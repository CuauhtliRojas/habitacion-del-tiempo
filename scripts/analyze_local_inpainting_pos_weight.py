from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcula la fraccion promedio de pixeles blancos y el pos_weight "
            "sugerido para mascaras binarias de local_inpainting."
        )
    )
    parser.add_argument(
        "--mask-dir",
        default="Dataset_U-Net_dual_decoder_fs_li/local_inpainting/train/mascara_fake",
        help="Directorio de mascaras fake ground truth.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=127,
        help="Umbral uint8 para considerar un pixel como positivo/blanco.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Procesos paralelos. Usa 0 para ejecucion secuencial.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/diagnostics/local_inpainting_pos_weight.json",
        help="Ruta de salida JSON con el resumen.",
    )
    return parser.parse_args()


def find_mask_paths(mask_dir: Path) -> list[Path]:
    if not mask_dir.exists():
        raise FileNotFoundError(f"No existe el directorio de mascaras: {mask_dir}")

    paths = [
        path
        for path in mask_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]

    if not paths:
        raise RuntimeError(f"No se encontraron mascaras en: {mask_dir}")

    return sorted(paths)


def analyze_mask(path_threshold: tuple[str, int]) -> dict[str, int | str]:
    path_str, threshold = path_threshold
    path = Path(path_str)

    with Image.open(path) as image:
        mask = image.convert("L")
        array = np.asarray(mask, dtype=np.uint8)

    positive_pixels = int((array > threshold).sum())
    total_pixels = int(array.size)
    negative_pixels = total_pixels - positive_pixels

    return {
        "path": str(path),
        "positive_pixels": positive_pixels,
        "negative_pixels": negative_pixels,
        "total_pixels": total_pixels,
    }


def safe_pos_weight(negative_pixels: int, positive_pixels: int) -> float:
    if positive_pixels <= 0:
        return math.inf
    return negative_pixels / positive_pixels


def main() -> None:
    args = parse_args()

    mask_dir = Path(args.mask_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    mask_paths = find_mask_paths(mask_dir)
    jobs = [(str(path), int(args.threshold)) for path in mask_paths]

    if args.workers and args.workers > 0:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(analyze_mask, jobs))
    else:
        rows = [analyze_mask(job) for job in jobs]

    total_positive = int(sum(int(row["positive_pixels"]) for row in rows))
    total_negative = int(sum(int(row["negative_pixels"]) for row in rows))
    total_pixels = int(sum(int(row["total_pixels"]) for row in rows))

    if total_pixels <= 0:
        raise RuntimeError("No hay pixeles analizables.")

    global_positive_fraction = total_positive / total_pixels
    global_negative_fraction = total_negative / total_pixels
    suggested_pos_weight = safe_pos_weight(total_negative, total_positive)

    per_mask_positive_fraction = [
        int(row["positive_pixels"]) / int(row["total_pixels"])
        for row in rows
        if int(row["total_pixels"]) > 0
    ]

    summary = {
        "mask_dir": str(mask_dir),
        "threshold_uint8": int(args.threshold),
        "num_masks": len(rows),
        "total_pixels": total_pixels,
        "total_positive_pixels": total_positive,
        "total_negative_pixels": total_negative,
        "global_positive_fraction": global_positive_fraction,
        "global_negative_fraction": global_negative_fraction,
        "suggested_pos_weight_negative_over_positive": suggested_pos_weight,
        "per_mask_positive_fraction_mean": float(np.mean(per_mask_positive_fraction)),
        "per_mask_positive_fraction_median": float(np.median(per_mask_positive_fraction)),
        "per_mask_positive_fraction_p05": float(np.percentile(per_mask_positive_fraction, 5)),
        "per_mask_positive_fraction_p95": float(np.percentile(per_mask_positive_fraction, 95)),
    }

    output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== LOCAL INPAINTING POS_WEIGHT ANALYSIS ===")
    print(f"Mask dir: {mask_dir}")
    print(f"Masks analyzed: {summary['num_masks']}")
    print(f"Total pixels: {summary['total_pixels']}")
    print(f"Positive pixels: {summary['total_positive_pixels']}")
    print(f"Negative pixels: {summary['total_negative_pixels']}")
    print(f"Global positive fraction: {summary['global_positive_fraction']:.10f}")
    print(f"Global negative fraction: {summary['global_negative_fraction']:.10f}")
    print(
        "Suggested pos_weight negative/positive: "
        f"{summary['suggested_pos_weight_negative_over_positive']:.10f}"
    )
    print(f"Per-mask positive fraction mean: {summary['per_mask_positive_fraction_mean']:.10f}")
    print(f"Per-mask positive fraction median: {summary['per_mask_positive_fraction_median']:.10f}")
    print(f"Per-mask positive fraction p05: {summary['per_mask_positive_fraction_p05']:.10f}")
    print(f"Per-mask positive fraction p95: {summary['per_mask_positive_fraction_p95']:.10f}")
    print(f"JSON: {output_json}")


if __name__ == "__main__":
    main()