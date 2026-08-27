"""
run_robustness_threshold.py

EXP-006: Gaussian robustness failure-threshold evaluation
for HLI-01 v1.0.0.

Reuses the validated EXP-005 robustness functions and
extends the perturbation range to identify the onset of
performance degradation.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import torch

import run_input_robustness as exp005


NOISE_FRACTIONS = [
    0.00,
    0.40,
    0.60,
    0.80,
    1.00,
    1.50,
    2.00,
]

OUTPUT_ROOT = Path("outputs/experiments")


def calculate_statistics(values):
    if len(values) == 1:
        return {
            "mean": values[0],
            "std": 0.0,
        }

    return {
        "mean": mean(values),
        "std": stdev(values),
    }


def save_results(run_results, aggregate_results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        OUTPUT_ROOT
        / f"EXP_{timestamp}_exp006_robustness_threshold"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_csv = output_dir / "exp006_threshold_runs.csv"

    with run_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "noise_fraction",
                "noise_sigma",
                "noise_seed",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ],
        )

        writer.writeheader()

        for row in run_results:
            writer.writerow(
                {
                    "noise_fraction": row["noise_fraction"],
                    "noise_sigma": row["noise_sigma"],
                    "noise_seed": row["noise_seed"],
                    "accuracy": row["accuracy"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1_score": row["f1_score"],
                }
            )

    summary_json = output_dir / "exp006_threshold_summary.json"

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            aggregate_results,
            file,
            indent=4,
        )

    return output_dir


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("EXP-006 Robustness Failure Threshold")
    print("=" * 60)

    print()
    print("Device              :", device)
    print(
        "Reference checkpoint:",
        exp005.REFERENCE_CHECKPOINT,
    )

    model = exp005.build_model(device)
    test_loader = exp005.get_test_loader()

    input_std = exp005.calculate_input_std(
        test_loader
    )

    print(
        "Empirical input std :",
        f"{input_std:.6f}",
    )

    run_results = []

    for noise_fraction in NOISE_FRACTIONS:

        noise_sigma = (
            noise_fraction
            * input_std
        )

        seeds = (
            [0]
            if noise_fraction == 0.0
            else exp005.NOISE_SEEDS
        )

        print()
        print("-" * 60)
        print(
            "Noise fraction:",
            f"{noise_fraction:.2f}",
        )
        print(
            "Noise sigma   :",
            f"{noise_sigma:.6f}",
        )
        print("-" * 60)

        for noise_seed in seeds:

            result = exp005.evaluate_once(
                model=model,
                test_loader=test_loader,
                device=device,
                noise_sigma=noise_sigma,
                noise_seed=noise_seed,
            )

            row = {
                "noise_fraction": noise_fraction,
                "noise_sigma": noise_sigma,
                "noise_seed": noise_seed,
                **result,
            }

            run_results.append(row)

            print(
                f"seed={noise_seed:>4} "
                f"accuracy={result['accuracy']:.4f} "
                f"f1={result['f1_score']:.4f}"
            )

    clean_result = next(
        row
        for row in run_results
        if row["noise_fraction"] == 0.0
    )

    clean_accuracy = clean_result["accuracy"]
    clean_f1 = clean_result["f1_score"]

    levels = []

    for noise_fraction in NOISE_FRACTIONS:

        subset = [
            row
            for row in run_results
            if row["noise_fraction"] == noise_fraction
        ]

        accuracy_values = [
            row["accuracy"]
            for row in subset
        ]

        precision_values = [
            row["precision"]
            for row in subset
        ]

        recall_values = [
            row["recall"]
            for row in subset
        ]

        f1_values = [
            row["f1_score"]
            for row in subset
        ]

        accuracy_stats = calculate_statistics(
            accuracy_values
        )

        f1_stats = calculate_statistics(
            f1_values
        )

        levels.append(
            {
                "noise_fraction": noise_fraction,
                "noise_sigma": subset[0]["noise_sigma"],
                "evaluations": len(subset),
                "accuracy": accuracy_stats,
                "precision": calculate_statistics(
                    precision_values
                ),
                "recall": calculate_statistics(
                    recall_values
                ),
                "f1_score": f1_stats,
                "accuracy_drop_from_clean":
                    clean_accuracy
                    - accuracy_stats["mean"],
                "f1_drop_from_clean":
                    clean_f1
                    - f1_stats["mean"],
            }
        )

    first_degradation = None

    for level in levels:
        if (
            level["noise_fraction"] > 0.0
            and level["accuracy"]["mean"] < clean_accuracy
        ):
            first_degradation = level
            break

    aggregate_results = {
        "experiment": "EXP-006",
        "description":
            "Gaussian robustness failure-threshold evaluation",
        "reference_model": {
            "architecture": "BiLSTM_Attention",
            "training_seed": exp005.REFERENCE_SEED,
            "checkpoint":
                str(exp005.REFERENCE_CHECKPOINT),
        },
        "input_standard_deviation": input_std,
        "noise_fractions": NOISE_FRACTIONS,
        "noise_seeds": exp005.NOISE_SEEDS,
        "number_of_evaluations": len(run_results),
        "first_observed_degradation":
            first_degradation,
        "levels": levels,
    }

    print()
    print("=" * 60)
    print("EXP-006 Threshold Summary")
    print("=" * 60)

    for level in levels:
        print()
        print(
            "Noise:",
            f"{level['noise_fraction'] * 100:.0f}% "
            f"(sigma={level['noise_sigma']:.6f})"
        )
        print(
            "Accuracy:",
            f"mean={level['accuracy']['mean']:.4f}, "
            f"std={level['accuracy']['std']:.4f}"
        )
        print(
            "F1-score:",
            f"mean={level['f1_score']['mean']:.4f}, "
            f"std={level['f1_score']['std']:.4f}"
        )
        print(
            "Accuracy drop:",
            f"{level['accuracy_drop_from_clean']:.4f}"
        )
        print(
            "F1 drop      :",
            f"{level['f1_drop_from_clean']:.4f}"
        )

    print()

    if first_degradation is None:
        print(
            "No accuracy degradation observed "
            "within the tested range."
        )
    else:
        print(
            "First observed mean accuracy degradation:"
        )
        print(
            f"{first_degradation['noise_fraction'] * 100:.0f}% "
            f"noise "
            f"(sigma={first_degradation['noise_sigma']:.6f})"
        )

    output_dir = save_results(
        run_results=run_results,
        aggregate_results=aggregate_results,
    )

    print()
    print("EXP-006 results saved to:")
    print(output_dir)

    print()
    print("Files:")
    print("  exp006_threshold_runs.csv")
    print("  exp006_threshold_summary.json")
    print("=" * 60)

    return aggregate_results


if __name__ == "__main__":
    main()
