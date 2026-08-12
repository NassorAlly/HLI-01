"""
compare_experiments.py

Compare saved HLI-01 research experiments.
"""

import argparse
import csv
import json
from pathlib import Path

from experiments.metrics_loader import ExperimentMetricsLoader
from src.evaluation.model_comparison import ModelComparison


def compare_experiments(
    experiments_dir="outputs/experiments",
    metric="test_accuracy",
):
    """
    Load and rank HLI-01 experiments by a selected metric.
    """

    records = ExperimentMetricsLoader.load(
        experiments_dir,
        metric=metric,
    )

    if not records:
        raise ValueError(
            f"No experiments contain metric '{metric}'."
        )

    return ModelComparison.compare(
        records,
        metric=metric,
    )


def save_comparison(results, output_dir, metric):
    """
    Save comparison results as JSON and CSV.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"comparison_{metric}.json"
    csv_path = output_dir / f"comparison_{metric}.csv"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    fieldnames = []

    for result in results:
        for key in result.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)

    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare HLI-01 experiment results."
    )

    parser.add_argument(
        "--metric",
        default="test_accuracy",
        help="Metric used to rank experiments.",
    )

    parser.add_argument(
        "--experiments-dir",
        default="outputs/experiments",
        help="Directory containing experiment runs.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/comparisons",
        help="Directory for comparison outputs.",
    )

    args = parser.parse_args()

    results = compare_experiments(
        experiments_dir=args.experiments_dir,
        metric=args.metric,
    )

    print("=" * 70)
    print("HLI-01 Experiment Comparison")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. {result['name']} "
            f"| {args.metric}="
            f"{result[args.metric]:.4f}"
        )

    json_path, csv_path = save_comparison(
        results,
        output_dir=args.output_dir,
        metric=args.metric,
    )

    print()
    print("Comparison JSON :", json_path)
    print("Comparison CSV  :", csv_path)


if __name__ == "__main__":
    main()
