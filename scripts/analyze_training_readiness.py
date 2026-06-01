from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import SegmentationSample, build_samples, count_by_attack, split_train_val


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("El archivo YAML debe contener un objeto de configuracion.")

    return data


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0

    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def load_binary_mask(path: Path, image_size: int | None) -> np.ndarray:
    mask = Image.open(path).convert("L")

    if image_size is not None:
        mask = mask.resize((image_size, image_size), Image.NEAREST)

    array = np.asarray(mask, dtype=np.float32) / 255.0
    return array > 0.5


def summarize_ratios(ratios: list[float], *, total_positive: int, total_pixels: int) -> dict[str, Any]:
    if total_pixels <= 0:
        estimated_pos_weight = None
    elif total_positive <= 0:
        estimated_pos_weight = math.inf
    else:
        estimated_pos_weight = (total_pixels - total_positive) / total_positive

    return {
        "count": len(ratios),
        "positive_ratio_mean": mean(ratios) if ratios else 0.0,
        "positive_ratio_median": median(ratios) if ratios else 0.0,
        "positive_ratio_min": min(ratios) if ratios else 0.0,
        "positive_ratio_max": max(ratios) if ratios else 0.0,
        "positive_ratio_p01": percentile(ratios, 1),
        "positive_ratio_p05": percentile(ratios, 5),
        "positive_ratio_p10": percentile(ratios, 10),
        "positive_ratio_p25": percentile(ratios, 25),
        "positive_ratio_p50": percentile(ratios, 50),
        "positive_ratio_p75": percentile(ratios, 75),
        "positive_ratio_p90": percentile(ratios, 90),
        "positive_ratio_p95": percentile(ratios, 95),
        "positive_ratio_p99": percentile(ratios, 99),
        "empty_masks": sum(1 for value in ratios if value == 0.0),
        "near_empty_masks_ratio_lt_0_001": sum(1 for value in ratios if 0.0 < value < 0.001),
        "tiny_masks_ratio_lt_0_01": sum(1 for value in ratios if 0.0 < value < 0.01),
        "large_masks_ratio_gt_0_25": sum(1 for value in ratios if value > 0.25),
        "total_positive_pixels": total_positive,
        "total_pixels": total_pixels,
        "estimated_pos_weight": estimated_pos_weight,
    }


def analyze_samples(
    *,
    phase: str,
    samples: Iterable[SegmentationSample],
    image_size: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fake_by_group: dict[str, list[float]] = {}
    authentic_by_group: dict[str, list[float]] = {}
    fake_positive_by_group: dict[str, int] = {}
    authentic_positive_by_group: dict[str, int] = {}
    pixels_by_group: dict[str, int] = {}

    for sample in samples:
        fake_mask = load_binary_mask(sample.fake_mask_path, image_size)
        authentic_mask = load_binary_mask(sample.authentic_mask_path, image_size)

        total_pixels = int(fake_mask.size)
        fake_positive = int(fake_mask.sum())
        authentic_positive = int(authentic_mask.sum())

        fake_ratio = fake_positive / total_pixels if total_pixels else 0.0
        authentic_ratio = authentic_positive / total_pixels if total_pixels else 0.0

        key = f"{phase}/{sample.attack}"

        fake_by_group.setdefault(key, []).append(fake_ratio)
        authentic_by_group.setdefault(key, []).append(authentic_ratio)
        fake_positive_by_group[key] = fake_positive_by_group.get(key, 0) + fake_positive
        authentic_positive_by_group[key] = authentic_positive_by_group.get(key, 0) + authentic_positive
        pixels_by_group[key] = pixels_by_group.get(key, 0) + total_pixels

        rows.append(
            {
                "phase": phase,
                "attack": sample.attack,
                "source_split": sample.split,
                "stem": sample.stem,
                "fake_positive_pixels": fake_positive,
                "fake_positive_ratio": fake_ratio,
                "authentic_positive_pixels": authentic_positive,
                "authentic_positive_ratio": authentic_ratio,
                "total_pixels": total_pixels,
                "fake_mask_path": str(sample.fake_mask_path),
                "authentic_mask_path": str(sample.authentic_mask_path),
            }
        )

    summary: dict[str, Any] = {}

    for key, ratios in fake_by_group.items():
        summary.setdefault(key, {})
        summary[key]["fake"] = summarize_ratios(
            ratios,
            total_positive=fake_positive_by_group[key],
            total_pixels=pixels_by_group[key],
        )
        summary[key]["authentic"] = summarize_ratios(
            authentic_by_group[key],
            total_positive=authentic_positive_by_group[key],
            total_pixels=pixels_by_group[key],
        )

    return rows, summary


def build_recommendation(summary: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    train_keys = [key for key in summary if key.startswith("train/")]
    train_fake_pos_weights = [
        summary[key]["fake"]["estimated_pos_weight"]
        for key in train_keys
        if summary[key]["fake"]["estimated_pos_weight"] not in (None, math.inf)
    ]

    if train_fake_pos_weights:
        estimated = float(mean(train_fake_pos_weights))
    else:
        estimated = None

    configured_pos_weight = float(config.get("pos_weight", 1.0))

    if estimated is None:
        pos_weight_note = "No fue posible estimar pos_weight porque no se encontraron pixeles positivos fake."
        suggested_pos_weight = configured_pos_weight
    else:
        suggested_pos_weight = max(1.0, min(200.0, estimated))
        if configured_pos_weight > estimated * 1.75:
            pos_weight_note = "El pos_weight configurado parece agresivo frente a la distribucion real; vigilar sobresegmentacion."
        elif configured_pos_weight < estimated * 0.5:
            pos_weight_note = "El pos_weight configurado parece bajo frente a la distribucion real; vigilar mascaras casi vacias."
        else:
            pos_weight_note = "El pos_weight configurado esta dentro de un rango razonable frente a la distribucion real."

    batch_size = int(config.get("batch_size", 1))
    epochs = int(config.get("epochs", 1))
    train_count = sum(summary[key]["fake"]["count"] for key in train_keys)

    return {
        "configured": {
            "experiment_name": config.get("experiment_name"),
            "image_size": config.get("image_size"),
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": config.get("learning_rate"),
            "pos_weight": configured_pos_weight,
            "threshold": config.get("threshold"),
            "mixed_precision": config.get("mixed_precision"),
        },
        "estimated_fake_pos_weight_from_train": estimated,
        "suggested_fake_pos_weight_clamped_1_200": suggested_pos_weight,
        "pos_weight_note": pos_weight_note,
        "batch_estimates": {
            "batch_1_train_batches_per_epoch": math.ceil(train_count / 1) if train_count else 0,
            "batch_2_train_batches_per_epoch": math.ceil(train_count / 2) if train_count else 0,
            "batch_4_train_batches_per_epoch": math.ceil(train_count / 4) if train_count else 0,
            "batch_8_train_batches_per_epoch": math.ceil(train_count / 8) if train_count else 0,
            "configured_total_train_updates": math.ceil(train_count / batch_size) * epochs if batch_size > 0 else 0,
        },
        "metric_priority": [
            "val_dice_fake",
            "val_iou_fake",
            "val_f1_fake",
            "val_precision_fake",
            "val_recall_fake",
            "val_accuracy_fake_solo_como_referencia",
        ],
    }


def write_report(
    *,
    path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    recommendation: dict[str, Any],
    train_counts: dict[str, int],
    val_counts: dict[str, int],
) -> None:
    lines: list[str] = []
    lines.append("=== DIAGNOSTICO PRE-ENTRENAMIENTO ===")
    lines.append(f"Experimento: {config.get('experiment_name')}")
    lines.append(f"Image size analizado: {config.get('image_size')}")
    lines.append(f"Batch size configurado: {config.get('batch_size')}")
    lines.append(f"Epochs configuradas: {config.get('epochs')}")
    lines.append(f"Learning rate configurado: {config.get('learning_rate')}")
    lines.append(f"Pos weight configurado: {config.get('pos_weight')}")
    lines.append("")
    lines.append("=== DISTRIBUCION TRAIN ===")
    lines.append(str(train_counts))
    lines.append("")
    lines.append("=== DISTRIBUCION VAL ===")
    lines.append(str(val_counts))
    lines.append("")

    for key in sorted(summary):
        lines.append(f"=== {key} ===")
        fake = summary[key]["fake"]
        lines.append("[fake]")
        lines.append(f"muestras: {fake['count']}")
        lines.append(f"ratio positivo mean: {fake['positive_ratio_mean']:.8f}")
        lines.append(f"ratio positivo median: {fake['positive_ratio_median']:.8f}")
        lines.append(f"p05/p50/p95: {fake['positive_ratio_p05']:.8f} / {fake['positive_ratio_p50']:.8f} / {fake['positive_ratio_p95']:.8f}")
        lines.append(f"mascaras vacias: {fake['empty_masks']}")
        lines.append(f"mascaras casi vacias <0.001: {fake['near_empty_masks_ratio_lt_0_001']}")
        lines.append(f"mascaras pequenas <0.01: {fake['tiny_masks_ratio_lt_0_01']}")
        lines.append(f"mascaras grandes >0.25: {fake['large_masks_ratio_gt_0_25']}")
        lines.append(f"pos_weight estimado: {fake['estimated_pos_weight']}")
        lines.append("")
        authentic = summary[key]["authentic"]
        lines.append("[authentic]")
        lines.append(f"ratio positivo mean: {authentic['positive_ratio_mean']:.8f}")
        lines.append(f"ratio positivo median: {authentic['positive_ratio_median']:.8f}")
        lines.append(f"p05/p50/p95: {authentic['positive_ratio_p05']:.8f} / {authentic['positive_ratio_p50']:.8f} / {authentic['positive_ratio_p95']:.8f}")
        lines.append(f"pos_weight estimado: {authentic['estimated_pos_weight']}")
        lines.append("")

    lines.append("=== RECOMENDACION AUTOMATICA ===")
    lines.append(json.dumps(recommendation, indent=2, ensure_ascii=False))
    lines.append("")
    lines.append("Nota: este diagnostico no entrena el modelo. Solo analiza mascaras, configuracion y conteos.")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera diagnostico pre-entrenamiento de mascaras, desbalance y trazabilidad."
    )
    parser.add_argument("--config", required=True, help="Ruta al YAML o config.resolved.json.")
    parser.add_argument("--output-dir", default=None, help="Directorio de salida. Por defecto usa outputs/diagnostics/<experiment_name>.")
    parser.add_argument("--image-size", type=int, default=None, help="Sobrescribe image_size solo para el diagnostico.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)

    image_size = args.image_size if args.image_size is not None else int(config["image_size"])
    config = dict(config)
    config["image_size"] = image_size

    experiment_name = str(config["experiment_name"])
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / "diagnostics" / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

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

    train_rows, train_summary = analyze_samples(
        phase="train",
        samples=train_samples,
        image_size=image_size,
    )
    val_rows, val_summary = analyze_samples(
        phase="val",
        samples=val_samples,
        image_size=image_size,
    )

    rows = train_rows + val_rows
    summary = {**train_summary, **val_summary}
    recommendation = build_recommendation(summary, config)

    csv_path = output_dir / "mask_distribution.csv"
    json_path = output_dir / "mask_distribution_summary.json"
    recommendation_path = output_dir / "training_readiness_recommendation.json"
    report_path = output_dir / "training_readiness_report.txt"
    config_path = output_dir / "diagnostic_config_snapshot.json"

    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    recommendation_path.write_text(json.dumps(recommendation, indent=2, ensure_ascii=False), encoding="utf-8")
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    write_report(
        path=report_path,
        config=config,
        summary=summary,
        recommendation=recommendation,
        train_counts=count_by_attack(train_samples),
        val_counts=count_by_attack(val_samples),
    )

    print("=== DIAGNOSTICO PRE-ENTRENAMIENTO GENERADO ===")
    print("Output dir:", output_dir)
    print("CSV:", csv_path)
    print("Summary JSON:", json_path)
    print("Recommendation JSON:", recommendation_path)
    print("Report:", report_path)


if __name__ == "__main__":
    main()
