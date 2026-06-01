from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crea README_RESUME.md dentro de un experimento para moverlo a otra PC o Colab."
    )
    parser.add_argument("experiment_dir", help="Ruta a outputs/experiments/<experiment_name>")
    return parser


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = Path(args.experiment_dir)

    config_path = experiment_dir / "config.resolved.json"
    manifest_path = experiment_dir / "run_manifest.json"
    last_checkpoint = experiment_dir / "checkpoints" / "last.pth"
    metrics_csv = experiment_dir / "metrics" / "metrics.csv"

    config = load_json(config_path)
    manifest = load_json(manifest_path)

    experiment_name = config.get("experiment_name", experiment_dir.name)
    git_commit = manifest.get("git", {}).get("commit", "desconocido")
    torch_version = manifest.get("torch", {}).get("version", "desconocido")
    cuda_runtime = manifest.get("torch", {}).get("cuda_runtime", "desconocido")

    content = f"""# Reanudacion del experimento: {experiment_name}

Este archivo resume como mover y reanudar el experimento en otra PC, otra instalacion local o Colab.

## Archivos minimos a conservar

- config.resolved.json
- run_manifest.json
- metrics/metrics.csv
- checkpoints/last.pth
- checkpoints/best_val_loss.pth, si existe
- checkpoints/best_val_dice.pth, si existe

## Estado tecnico registrado

- Commit Git: {git_commit}
- PyTorch: {torch_version}
- CUDA runtime PyTorch: {cuda_runtime}

## Checkpoint principal

Ruta esperada:
{last_checkpoint}

## Metricas

Ruta esperada:
{metrics_csv}

## Reanudar entrenamiento desde la raiz del repo

uv run python .\\train_orchestrator.py --config {config_path} --resume {last_checkpoint}

## Regenerar graficas y reporte sin entrenar

uv run python .\\scripts\\recompute_experiment_artifacts.py {experiment_dir}

## Inspeccionar checkpoint sin GPU

uv run python .\\scripts\\inspect_checkpoint.py {last_checkpoint}

## Nota

Para reanudar correctamente, el repo debe estar en el mismo commit o en un commit compatible con la arquitectura, nombres de capas, formato de checkpoint y contrato de metricas.
"""

    output_path = experiment_dir / "README_RESUME.md"
    output_path.write_text(content, encoding="utf-8")

    print("README_RESUME generado:", output_path)


if __name__ == "__main__":
    main()
