from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspecciona un checkpoint .pth sin cargarlo en GPU."
    )
    parser.add_argument("checkpoint", help="Ruta al checkpoint .pth")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print("=== CHECKPOINT INSPECTION ===")
    print("Path:", checkpoint_path)
    print("Keys:", sorted(checkpoint.keys()))

    epoch = checkpoint.get("epoch")
    config = checkpoint.get("config", {})
    metrics_history = checkpoint.get("metrics_history", [])

    print("Epoch:", epoch)
    print("Best val loss:", checkpoint.get("best_val_loss"))
    print("Best val dice:", checkpoint.get("best_val_dice"))
    print("Metrics history rows:", len(metrics_history))

    if config:
        print("\n=== CONFIG RESUMEN ===")
        for key in [
            "experiment_name",
            "dataset_root",
            "output_root",
            "attacks",
            "image_size",
            "batch_size",
            "epochs",
            "learning_rate",
            "pos_weight",
            "val_ratio",
            "threshold",
            "mixed_precision",
        ]:
            if key in config:
                print(f"{key}: {config[key]}")

        config_path = Path(config.get("output_root", "outputs/experiments")) / str(
            config.get("experiment_name", "<experiment_name>")
        ) / "config.resolved.json"

        print("\n=== COMANDO CONCEPTUAL DE RESUME ===")
        print(
            "uv run python .\\train_orchestrator.py `\n"
            f"  --config {config_path} `\n"
            f"  --resume {checkpoint_path}"
        )

    if metrics_history:
        print("\n=== ULTIMA FILA DE METRICAS ===")
        last = metrics_history[-1]
        for key, value in last.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
