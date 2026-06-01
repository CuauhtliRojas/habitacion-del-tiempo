from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plots import analyze_fit_status, generate_training_plots


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regenera graficas y fit_report desde metrics.csv sin reentrenar."
    )
    parser.add_argument("experiment_dir", help="Ruta a outputs/experiments/<experiment_name>")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_dir = Path(args.experiment_dir)

    metrics_csv = experiment_dir / "metrics" / "metrics.csv"
    plots_dir = experiment_dir / "plots"
    report_path = experiment_dir / "fit_report.txt"

    if not metrics_csv.exists():
        raise FileNotFoundError(f"No existe metrics.csv: {metrics_csv}")

    generated = generate_training_plots(
        metrics_csv=metrics_csv,
        output_dir=plots_dir,
    )

    report = analyze_fit_status(
        metrics_csv=metrics_csv,
        output_path=report_path,
    )

    print("=== RECOMPUTE EXPERIMENT ARTIFACTS ===")
    print("Experiment dir:", experiment_dir)
    print("Metrics:", metrics_csv)
    print("Plots dir:", plots_dir)
    print("Report:", report_path)
    print("Generated plots:")
    for path in generated:
        print("-", path)

    print("\n=== FIT REPORT ===")
    print(report)


if __name__ == "__main__":
    main()
