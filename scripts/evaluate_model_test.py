from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import DualMaskSegmentationDataset, build_samples
from src.metrics import (
    accuracy_score_mask,
    dice_score,
    iou_score,
    logits_to_probabilities,
    precision_score_mask,
    recall_score_mask,
    f1_score_mask,
    soft_dice_loss,
    weighted_bce_from_logits,
)
from src.model import DualSegmentationModel


"""
Evalúa un checkpoint en el split test y guarda overlays de error.

Colores del overlay:
- verde: acierto positivo.
- amarillo: falso negativo.
- rojo: falso positivo.

Referencias internas:
- src/model.py
- src/dataset.py
- src/metrics.py
- src/checkpoints.py

Referencia PyTorch:
https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
"""


EPS = 1e-7


def load_config(path: str | Path) -> dict[str, Any]:
    """
    Lee el YAML usado para el experimento.
    """
    with open(path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("El YAML debe contener un objeto de configuración.")

    return config


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calcula IoU loss solo para reporte.

    El entrenamiento principal usa BCE + Dice. Aquí IoU loss se imprime para
    compararlo con resultados tipo test legacy.
    """
    pred = logits_to_probabilities(logits)
    target = target.float()

    intersection = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

    iou = (intersection + EPS) / (union + EPS)
    return 1.0 - iou.mean()


def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device) -> DualSegmentationModel:
    """
    Carga pesos desde un checkpoint completo o desde un state_dict simple.
    """
    model = DualSegmentationModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def tensor_image_to_uint8(image: torch.Tensor) -> np.ndarray:
    """
    Convierte imagen [C, H, W] en uint8 [H, W, 3].
    """
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    array = image.numpy()
    array = np.transpose(array, (1, 2, 0))
    return (array * 255.0).astype(np.uint8)


def mask_to_uint8(mask: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """
    Convierte una máscara a blanco y negro.
    """
    mask = mask.detach().cpu().float()

    if mask.ndim == 3:
        mask = mask[0]

    return (mask.numpy() >= threshold).astype(np.uint8) * 255


def make_error_overlay(
    *,
    image: torch.Tensor,
    target_mask: torch.Tensor,
    pred_prob: torch.Tensor,
    threshold: float,
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Genera overlay de error.

    verde = verdadero positivo.
    amarillo = falso negativo.
    rojo = falso positivo.
    """
    image_uint8 = tensor_image_to_uint8(image).astype(np.float32)

    target = target_mask.detach().cpu().float()
    pred = pred_prob.detach().cpu().float()

    if target.ndim == 3:
        target = target[0]

    if pred.ndim == 3:
        pred = pred[0]

    target_bool = target.numpy() >= 0.5
    pred_bool = pred.numpy() >= threshold

    true_positive = pred_bool & target_bool
    false_negative = (~pred_bool) & target_bool
    false_positive = pred_bool & (~target_bool)

    color = np.zeros_like(image_uint8)

    color[true_positive] = np.array([0, 255, 0], dtype=np.float32)
    color[false_negative] = np.array([255, 255, 0], dtype=np.float32)
    color[false_positive] = np.array([255, 0, 0], dtype=np.float32)

    error_mask = true_positive | false_negative | false_positive

    overlay = image_uint8.copy()
    overlay[error_mask] = (
        (1.0 - alpha) * overlay[error_mask] + alpha * color[error_mask]
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_overlay_sample(
    *,
    output_dir: Path,
    stem: str,
    image: torch.Tensor,
    fake_target: torch.Tensor,
    fake_prob: torch.Tensor,
    authentic_target: torch.Tensor,
    authentic_prob: torch.Tensor,
    threshold: float,
) -> None:
    """
    Guarda overlays y máscaras de una muestra.
    """
    sample_dir = output_dir / stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    image_uint8 = tensor_image_to_uint8(image)

    fake_pred_uint8 = mask_to_uint8(fake_prob, threshold)
    fake_gt_uint8 = mask_to_uint8(fake_target, 0.5)
    authentic_pred_uint8 = mask_to_uint8(authentic_prob, threshold)
    authentic_gt_uint8 = mask_to_uint8(authentic_target, 0.5)

    fake_overlay = make_error_overlay(
        image=image,
        target_mask=fake_target,
        pred_prob=fake_prob,
        threshold=threshold,
    )

    authentic_overlay = make_error_overlay(
        image=image,
        target_mask=authentic_target,
        pred_prob=authentic_prob,
        threshold=threshold,
    )

    Image.fromarray(image_uint8).save(sample_dir / "image.png")

    Image.fromarray(fake_gt_uint8).save(sample_dir / "fake_gt.png")
    Image.fromarray(fake_pred_uint8).save(sample_dir / "fake_pred.png")
    Image.fromarray(fake_overlay).save(sample_dir / "fake_overlay_error.png")

    Image.fromarray(authentic_gt_uint8).save(sample_dir / "authentic_gt.png")
    Image.fromarray(authentic_pred_uint8).save(sample_dir / "authentic_pred.png")
    Image.fromarray(authentic_overlay).save(sample_dir / "authentic_overlay_error.png")


def empty_stats() -> dict[str, float]:
    """
    Crea acumulador de métricas.
    """
    return {
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "dice_score": 0.0,
        "iou_score": 0.0,
        "bce_loss": 0.0,
        "dice_loss": 0.0,
        "iou_loss": 0.0,
        "total_loss": 0.0,
    }


def add_weighted(stats: dict[str, float], values: dict[str, float], weight: int) -> None:
    """
    Suma métricas ponderadas por tamaño de batch.
    """
    for key, value in values.items():
        stats[key] += float(value) * weight


def finalize_stats(stats: dict[str, float], total_samples: int) -> dict[str, float]:
    """
    Convierte sumas acumuladas en promedios.
    """
    return {
        key: value / max(total_samples, 1)
        for key, value in stats.items()
    }


def compute_mask_values(
    *,
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float,
    threshold: float,
) -> dict[str, float]:
    """
    Calcula métricas y pérdidas para una máscara.
    """
    prob = logits_to_probabilities(logits)

    bce = weighted_bce_from_logits(logits, target, pos_weight=pos_weight)
    dice_loss_value = soft_dice_loss(logits, target)
    iou_loss_value = soft_iou_loss(logits, target)

    return {
        "accuracy": accuracy_score_mask(prob, target, threshold),
        "precision": precision_score_mask(prob, target, threshold),
        "recall": recall_score_mask(prob, target, threshold),
        "f1": f1_score_mask(prob, target, threshold),
        "dice_score": dice_score(prob, target, threshold),
        "iou_score": iou_score(prob, target, threshold),
        "bce_loss": float(bce.detach().cpu()),
        "dice_loss": float(dice_loss_value.detach().cpu()),
        "iou_loss": float(iou_loss_value.detach().cpu()),
        "total_loss": float((bce + dice_loss_value + iou_loss_value).detach().cpu()),
    }


def print_report(title: str, stats: dict[str, float]) -> None:
    """
    Imprime resultados con formato simple.
    """
    print("=" * 34)
    print(title)
    print("=" * 34)
    print(f"accuracy: {stats['accuracy']:.4f} | porcentaje acierto: {stats['accuracy'] * 100:.2f}%")
    print(f"precision: {stats['precision']:.4f}")
    print(f"recall: {stats['recall']:.4f}")
    print(f"f1: {stats['f1']:.4f}")
    print(f"dice_score: {stats['dice_score']:.4f}")
    print(f"iou_score: {stats['iou_score']:.4f}")
    print(f"bce_loss: {stats['bce_loss']:.4f}")
    print(f"dice_loss: {stats['dice_loss']:.4f}")
    print(f"iou_loss: {stats['iou_loss']:.4f}")
    print(f"total_loss: {stats['total_loss']:.4f}")
    print("")


def save_metrics_csv(path: Path, fake_stats: dict[str, float], authentic_stats: dict[str, float]) -> None:
    """
    Guarda métricas en CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {"mask": "fake", **fake_stats},
        {"mask": "authentic", **authentic_stats},
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_json(
    path: Path,
    *,
    checkpoint: str,
    config: str,
    split: str,
    total_samples: int,
    fake_stats: dict[str, float],
    authentic_stats: dict[str, float],
) -> None:
    """
    Guarda resumen legible del test.
    """
    payload = {
        "checkpoint": checkpoint,
        "config": config,
        "split": split,
        "total_samples": total_samples,
        "fake": fake_stats,
        "authentic": authentic_stats,
    }

    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> None:
    """
    Ejecuta evaluación completa en test.
    """
    config = load_config(args.config)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    threshold = float(args.threshold if args.threshold is not None else config.get("threshold", 0.5))
    pos_weight = float(args.pos_weight if args.pos_weight is not None else config["pos_weight"])

    attacks = args.attacks or config["attacks"]
    split = args.split

    samples = build_samples(
        dataset_root=config["dataset_root"],
        attacks=attacks,
        split=split,
    )

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    dataset = DualMaskSegmentationDataset(
        samples=samples,
        image_size=int(config["image_size"]),
    )

    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or config["batch_size"]),
        shuffle=False,
        num_workers=int(args.num_workers if args.num_workers is not None else config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    checkpoint_path = Path(args.checkpoint)
    checkpoint_name = checkpoint_path.stem

    output_root = Path(args.output_dir)
    output_dir = output_root / checkpoint_name
    overlays_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_from_checkpoint(checkpoint_path, device)

    fake_stats = empty_stats()
    authentic_stats = empty_stats()

    total_samples = 0
    saved_overlays = 0

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split}")
    print(f"Ataques: {', '.join(attacks)}")
    print(f"Device: {device}")
    print(f"Threshold: {threshold}")
    print(f"Pos weight: {pos_weight}")
    print("")

    for batch in tqdm(loader, desc="test", unit="batch"):
        images = batch["image"].to(device)
        fake_masks = batch["fake_mask"].to(device)
        authentic_masks = batch["authentic_mask"].to(device)
        stems = batch["stem"]

        pred_fake_logits, pred_authentic_logits = model(images)

        batch_size = images.shape[0]
        total_samples += batch_size

        fake_values = compute_mask_values(
            logits=pred_fake_logits,
            target=fake_masks,
            pos_weight=pos_weight,
            threshold=threshold,
        )

        authentic_values = compute_mask_values(
            logits=pred_authentic_logits,
            target=authentic_masks,
            pos_weight=pos_weight,
            threshold=threshold,
        )

        add_weighted(fake_stats, fake_values, batch_size)
        add_weighted(authentic_stats, authentic_values, batch_size)

        if saved_overlays < args.max_overlays:
            fake_probs = logits_to_probabilities(pred_fake_logits)
            authentic_probs = logits_to_probabilities(pred_authentic_logits)

            for index in range(batch_size):
                if saved_overlays >= args.max_overlays:
                    break

                save_overlay_sample(
                    output_dir=overlays_dir,
                    stem=str(stems[index]),
                    image=images[index],
                    fake_target=fake_masks[index],
                    fake_prob=fake_probs[index],
                    authentic_target=authentic_masks[index],
                    authentic_prob=authentic_probs[index],
                    threshold=threshold,
                )

                saved_overlays += 1

    fake_final = finalize_stats(fake_stats, total_samples)
    authentic_final = finalize_stats(authentic_stats, total_samples)

    print_report("RESULTADOS TEST - FAKE", fake_final)
    print_report("RESULTADOS TEST - AUTÉNTICA", authentic_final)

    save_metrics_csv(output_dir / "metrics_test.csv", fake_final, authentic_final)
    save_summary_json(
        output_dir / "summary_test.json",
        checkpoint=str(checkpoint_path),
        config=str(args.config),
        split=split,
        total_samples=total_samples,
        fake_stats=fake_final,
        authentic_stats=authentic_final,
    )

    print("=" * 34)
    print(f"Muestras evaluadas: {total_samples}")
    print(f"Overlays guardados: {saved_overlays}")
    print(f"Salida: {output_dir}")
    print("=" * 34)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evalúa un checkpoint en test y genera overlays de error."
    )

    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--split", default="test")
    parser.add_argument("--attacks", nargs="+")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-overlays", type=int, default=32)
    parser.add_argument("--device")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()