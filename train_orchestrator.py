from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from Modelo_U_Net_dual_decoder import DualSegmentationModel
from src.checkpoints import load_checkpoint, save_checkpoint
from src.dataset import (
    DualMaskSegmentationDataset,
    build_samples,
    count_by_attack,
    limit_samples,
    split_train_val,
)
from src.metrics import dual_segmentation_loss, segmentation_metric_row
from src.plots import analyze_fit_status, generate_training_plots
from src.visualization import save_epoch_samples


def parse_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "y", "si", "sí"}:
        return True

    if normalized in {"0", "false", "no", "n"}:
        return False

    raise argparse.ArgumentTypeError(f"Valor booleano invalido: {value}")


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("El archivo YAML debe contener un objeto de configuracion.")

    return data


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    config = dict(config)

    overrides = {
        "experiment_name": args.experiment_name,
        "dataset_root": args.dataset_root,
        "output_root": args.output_root,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "pos_weight": args.pos_weight,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "threshold": args.threshold,
        "checkpoint_every": args.checkpoint_every,
        "sample_every": args.sample_every,
        "mixed_precision": args.mixed_precision,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "max_test_samples": args.max_test_samples,
    }

    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if args.attacks:
        config["attacks"] = args.attacks

    if args.resume:
        config["resume"] = args.resume

    if args.weights:
        config["weights"] = args.weights

    return config


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gpu_info(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": None,
            "total_vram_gb": None,
        }

    props = torch.cuda.get_device_properties(0)
    return {
        "device": str(device),
        "cuda_available": True,
        "gpu_name": props.name,
        "total_vram_gb": round(props.total_memory / (1024 ** 3), 2),
    }


def print_preflight(config: dict[str, Any], gpu_info: dict[str, Any], train_count: int, val_count: int) -> None:
    batch_size = int(config["batch_size"])
    image_size = int(config["image_size"])
    epochs = int(config["epochs"])

    train_batches = math.ceil(train_count / batch_size) if batch_size > 0 else 0
    val_batches = math.ceil(val_count / batch_size) if batch_size > 0 else 0

    print("\n=== PREFLIGHT DEL EXPERIMENTO ===")
    print(f"Experimento: {config['experiment_name']}")
    print(f"Dataset: {config['dataset_root']}")
    print(f"Ataques: {', '.join(config['attacks'])}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Train samples: {train_count}")
    print(f"Val samples: {val_count}")
    print(f"Train batches/epoch aprox: {train_batches}")
    print(f"Val batches/epoch aprox: {val_batches}")
    print(f"Total train batches aprox: {train_batches * epochs}")
    print(f"Device: {gpu_info['device']}")
    print(f"CUDA disponible: {gpu_info['cuda_available']}")
    print(f"GPU: {gpu_info['gpu_name']}")
    print(f"VRAM total GB: {gpu_info['total_vram_gb']}")

    warnings = []

    if image_size >= 512 and batch_size > 2:
        warnings.append(
            "512x512 con batch_size mayor a 2 puede exceder VRAM en GPU de 6GB."
        )

    if image_size >= 512 and gpu_info["total_vram_gb"] is not None and gpu_info["total_vram_gb"] <= 8:
        warnings.append(
            "512x512 en GPU de 6-8GB debe probarse primero con pocas muestras."
        )

    if train_count > 10000 and epochs >= 20:
        warnings.append(
            "Entrenamiento grande detectado: guarda checkpoints por epoca y evita cerrar la sesion."
        )

    if warnings:
        print("\n=== ADVERTENCIAS ===")
        for warning in warnings:
            print(f"- {warning}")

    print("")


def confirm_or_exit(args: argparse.Namespace) -> None:
    if args.yes or args.dry_run:
        return

    response = input("Continuar con esta configuracion? [s/N]: ").strip().lower()

    if response not in {"s", "si", "sí", "y", "yes"}:
        print("Cancelado por el usuario.")
        sys.exit(1)


def make_experiment_dirs(config: dict[str, Any]) -> dict[str, Path]:
    experiment_root = Path(config["output_root"]) / config["experiment_name"]

    dirs = {
        "root": experiment_root,
        "checkpoints": experiment_root / "checkpoints",
        "metrics": experiment_root / "metrics",
        "plots": experiment_root / "plots",
        "samples": experiment_root / "samples",
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs


def write_config_snapshot(config: dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    exists = path.exists()

    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def average_dicts(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}

    keys = rows[0].keys()
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in keys
    }


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    config: dict[str, Any],
    phase: str,
    scaler: torch.cuda.amp.GradScaler | None,
) -> dict[str, Any]:
    is_train = phase == "train"

    if is_train:
        model.train()
    else:
        model.eval()

    loss_rows: list[dict[str, float]] = []
    metric_rows: list[dict[str, float]] = []
    per_attack_rows: dict[str, list[dict[str, float]]] = {}

    progress = tqdm(loader, desc=phase, unit="batch")

    for batch in progress:
        images = batch["image"].to(device)
        fake_masks = batch["fake_mask"].to(device)
        authentic_masks = batch["authentic_mask"].to(device)
        attacks = batch["attack"]

        if is_train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        use_amp = bool(config.get("mixed_precision", False)) and device.type == "cuda"

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_fake, pred_authentic = model(images)

                loss, loss_parts = dual_segmentation_loss(
                    pred_fake=pred_fake,
                    target_fake=fake_masks,
                    pred_authentic=pred_authentic,
                    target_authentic=authentic_masks,
                    pos_weight=float(config["pos_weight"]),
                )

            if is_train:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        loss_rows.append(loss_parts)

        metrics = segmentation_metric_row(
            pred_fake=pred_fake.detach(),
            target_fake=fake_masks.detach(),
            pred_authentic=pred_authentic.detach(),
            target_authentic=authentic_masks.detach(),
            threshold=float(config["threshold"]),
        )
        metric_rows.append(metrics)

        for attack in set(attacks):
            attack_indices = [i for i, value in enumerate(attacks) if value == attack]
            idx_tensor = torch.tensor(attack_indices, dtype=torch.long, device=device)

            attack_metrics = segmentation_metric_row(
                pred_fake=pred_fake.detach().index_select(0, idx_tensor),
                target_fake=fake_masks.detach().index_select(0, idx_tensor),
                pred_authentic=pred_authentic.detach().index_select(0, idx_tensor),
                target_authentic=authentic_masks.detach().index_select(0, idx_tensor),
                threshold=float(config["threshold"]),
            )
            per_attack_rows.setdefault(attack, []).append(attack_metrics)

        progress.set_postfix(loss=f"{loss_parts['loss_total']:.4f}")

    avg_loss = average_dicts(loss_rows)
    avg_metrics = average_dicts(metric_rows)

    result: dict[str, Any] = {
        **avg_loss,
        **avg_metrics,
    }

    for attack, rows in per_attack_rows.items():
        attack_avg = average_dicts(rows)
        for key, value in attack_avg.items():
            result[f"{attack}_{key}"] = value

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orquestador de entrenamiento U-Net dual-decoder para segmentacion facial."
    )

    parser.add_argument("--config", required=True, help="Ruta al YAML de configuracion.")
    parser.add_argument("--experiment-name")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-root")
    parser.add_argument("--attacks", nargs="+")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--sample-every", type=int)
    parser.add_argument("--mixed-precision", type=parse_bool)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--weights")
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args)

    required_keys = [
        "experiment_name",
        "dataset_root",
        "output_root",
        "attacks",
        "train_split",
        "image_size",
        "batch_size",
        "epochs",
        "learning_rate",
        "pos_weight",
        "val_ratio",
        "seed",
        "num_workers",
        "threshold",
        "checkpoint_every",
        "mixed_precision",
    ]

    for key in required_keys:
        if key not in config:
            raise KeyError(f"Falta la clave requerida en config: {key}")

    torch.manual_seed(int(config["seed"]))

    device = get_device(args.device)
    gpu_info = get_gpu_info(device)

    all_train_samples = build_samples(
        dataset_root=config["dataset_root"],
        attacks=config["attacks"],
        split=config["train_split"],
    )

    train_samples, val_samples = split_train_val(
        all_train_samples,
        val_ratio=float(config["val_ratio"]),
        seed=int(config["seed"]),
        max_train_samples=config.get("max_train_samples"),
        max_val_samples=config.get("max_val_samples"),
    )

    print_preflight(config, gpu_info, len(train_samples), len(val_samples))
    print("Distribucion train:", count_by_attack(train_samples))
    print("Distribucion val:", count_by_attack(val_samples))
    print("")

    if args.dry_run:
        print("Dry-run finalizado. No se entreno el modelo.")
        return

    confirm_or_exit(args)

    dirs = make_experiment_dirs(config)
    write_config_snapshot(config, dirs["root"] / "config.resolved.json")

    train_dataset = DualMaskSegmentationDataset(train_samples, image_size=int(config["image_size"]))
    val_dataset = DualMaskSegmentationDataset(val_samples, image_size=int(config["image_size"]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model = DualSegmentationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))

    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.get("mixed_precision", False)) and device.type == "cuda"
    )

    start_epoch = 1
    best_val_loss: float | None = None
    best_val_dice: float | None = None
    metrics_history: list[dict[str, Any]] = []

    if config.get("weights"):
        checkpoint = torch.load(config["weights"], map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Pesos cargados desde: {config['weights']}")

    if config.get("resume"):
        checkpoint = load_checkpoint(
            path=config["resume"],
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )

        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = checkpoint.get("best_val_loss")
        best_val_dice = checkpoint.get("best_val_dice")
        metrics_history = checkpoint.get("metrics_history", [])
        print(f"Entrenamiento reanudado desde epoch {start_epoch}")

    metrics_csv = dirs["metrics"] / "metrics.csv"

    for epoch in range(start_epoch, int(config["epochs"]) + 1):
        print(f"\n=== Epoch {epoch}/{config['epochs']} ===")

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            config=config,
            phase="train",
            scaler=scaler,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            config=config,
            phase="val",
            scaler=None,
        )

        row = {
            "epoch": epoch,
            "train_loss_total": train_metrics.get("loss_total"),
            "train_loss_fake": train_metrics.get("loss_fake"),
            "train_loss_authentic": train_metrics.get("loss_authentic"),
            "train_dice_fake": train_metrics.get("dice_fake"),
            "train_iou_fake": train_metrics.get("iou_fake"),
            "val_loss_total": val_metrics.get("loss_total"),
            "val_loss_fake": val_metrics.get("loss_fake"),
            "val_loss_authentic": val_metrics.get("loss_authentic"),
            "val_dice_fake": val_metrics.get("dice_fake"),
            "val_iou_fake": val_metrics.get("iou_fake"),
            "val_precision_fake": val_metrics.get("precision_fake"),
            "val_recall_fake": val_metrics.get("recall_fake"),
            "val_f1_fake": val_metrics.get("f1_fake"),
        }

        for key, value in val_metrics.items():
            if key.startswith("faceswap_") or key.startswith("local_inpainting_"):
                row[f"val_{key}"] = value

        metrics_history.append(row)
        append_metrics_csv(metrics_csv, row)

        generated_plots = generate_training_plots(
            metrics_csv=metrics_csv,
            output_dir=dirs["plots"],
        )

        fit_report = analyze_fit_status(
            metrics_csv=metrics_csv,
            output_path=dirs["root"] / "fit_report.txt",
        )

        sample_every = int(config.get("sample_every", 0) or 0)
        if sample_every > 0 and epoch % sample_every == 0:
            save_epoch_samples(
                model=model,
                loader=val_loader,
                device=device,
                output_dir=dirs["samples"],
                epoch=epoch,
                threshold=float(config["threshold"]),
                max_samples=8,
            )

        print(
            f"Epoch {epoch}: "
            f"train_loss={row['train_loss_total']:.4f} "
            f"val_loss={row['val_loss_total']:.4f} "
            f"val_dice_fake={row['val_dice_fake']:.4f} "
            f"val_iou_fake={row['val_iou_fake']:.4f}"
        )

        checkpoint_every = int(config["checkpoint_every"])

        if checkpoint_every > 0 and epoch % checkpoint_every == 0:
            save_checkpoint(
                path=dirs["checkpoints"] / "last.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics_history=metrics_history,
                best_val_loss=best_val_loss,
                best_val_dice=best_val_dice,
                scaler=scaler,
            )

            save_checkpoint(
                path=dirs["checkpoints"] / f"epoch_{epoch:03d}.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics_history=metrics_history,
                best_val_loss=best_val_loss,
                best_val_dice=best_val_dice,
                scaler=scaler,
            )

        current_val_loss = float(row["val_loss_total"])
        current_val_dice = float(row["val_dice_fake"])

        if best_val_loss is None or current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            save_checkpoint(
                path=dirs["checkpoints"] / "best_val_loss.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics_history=metrics_history,
                best_val_loss=best_val_loss,
                best_val_dice=best_val_dice,
                scaler=scaler,
            )

        if best_val_dice is None or current_val_dice > best_val_dice:
            best_val_dice = current_val_dice
            save_checkpoint(
                path=dirs["checkpoints"] / "best_val_dice.pth",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                config=config,
                metrics_history=metrics_history,
                best_val_loss=best_val_loss,
                best_val_dice=best_val_dice,
                scaler=scaler,
            )

    print("\nEntrenamiento finalizado.")
    print(f"Metricas: {metrics_csv}")
    print(f"Checkpoints: {dirs['checkpoints']}")
    print(f"Graficas: {dirs['plots']}")
    print(f"Muestras visuales: {dirs['samples']}")
    print(f"Reporte de ajuste: {dirs['root'] / 'fit_report.txt'}")


if __name__ == "__main__":
    main()
