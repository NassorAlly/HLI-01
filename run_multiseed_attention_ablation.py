"""
run_multiseed_attention_ablation.py

EXP-004: Multi-seed paired attention ablation for HLI-01 v1.0.0.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

from run_experiment import initialize_experiment
from train import run_training


SEEDS = [42, 7, 21, 84, 123]

OUTPUT_ROOT = Path("outputs/experiments")


def run_variant(seed, use_attention):
    name = (
        f"exp004_seed_{seed}_"
        f"{'attention_on' if use_attention else 'attention_off'}"
    )

    experiment_dir, logger = initialize_experiment(
        experiment_name=name,
        seed=seed,
    )

    checkpoint_dir = experiment_dir / "checkpoints"
    figure_dir = experiment_dir / "figures"

    history, summary = run_training(
        checkpoint_dir=checkpoint_dir,
        figure_dir=figure_dir,
        seed=seed,
        use_attention=use_attention,
    )

    logger.save_training_history(history)
    logger.save_metrics(summary)

    return experiment_dir, summary


def calculate_statistics(values):
    return {
        "mean": mean(values),
        "std": stdev(values),
    }


def save_results(results, paired_results, aggregate_summary):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = (
        OUTPUT_ROOT
        / f"EXP_{timestamp}_exp004_attention_ablation_summary"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # Per-run CSV
    # --------------------------------------------------

    run_csv = output_dir / "exp004_runs.csv"

    with run_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "use_attention",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1_score",
                "experiment_dir",
            ],
        )

        writer.writeheader()

        for result in results:
            summary = result["summary"]

            writer.writerow(
                {
                    "seed": result["seed"],
                    "use_attention": result["use_attention"],
                    "test_accuracy": summary["test_accuracy"],
                    "test_precision": summary["test_precision"],
                    "test_recall": summary["test_recall"],
                    "test_f1_score": summary["test_f1_score"],
                    "experiment_dir": result["experiment_dir"],
                }
            )

    # --------------------------------------------------
    # Paired seed CSV
    # --------------------------------------------------

    paired_csv = output_dir / "exp004_paired_deltas.csv"

    with paired_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "accuracy_attention_on",
                "accuracy_attention_off",
                "accuracy_delta",
                "f1_attention_on",
                "f1_attention_off",
                "f1_delta",
            ],
        )

        writer.writeheader()
        writer.writerows(paired_results)

    # --------------------------------------------------
    # Aggregate JSON
    # --------------------------------------------------

    summary_json = output_dir / "exp004_summary.json"

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            aggregate_summary,
            file,
            indent=4,
        )

    return output_dir


def main():
    results = []

    # --------------------------------------------------
    # Run all paired variants
    # --------------------------------------------------

    for seed in SEEDS:
        for use_attention in (True, False):
            experiment_dir, summary = run_variant(
                seed=seed,
                use_attention=use_attention,
            )

            results.append(
                {
                    "seed": seed,
                    "use_attention": use_attention,
                    "experiment_dir": str(experiment_dir),
                    "summary": summary,
                }
            )

    # --------------------------------------------------
    # Separate Attention ON / OFF results
    # --------------------------------------------------

    attention_on = {
        result["seed"]: result
        for result in results
        if result["use_attention"]
    }

    attention_off = {
        result["seed"]: result
        for result in results
        if not result["use_attention"]
    }

    on_accuracy = [
        attention_on[seed]["summary"]["test_accuracy"]
        for seed in SEEDS
    ]

    off_accuracy = [
        attention_off[seed]["summary"]["test_accuracy"]
        for seed in SEEDS
    ]

    on_f1 = [
        attention_on[seed]["summary"]["test_f1_score"]
        for seed in SEEDS
    ]

    off_f1 = [
        attention_off[seed]["summary"]["test_f1_score"]
        for seed in SEEDS
    ]

    # --------------------------------------------------
    # Paired differences
    # --------------------------------------------------

    paired_results = []

    for seed in SEEDS:
        on_summary = attention_on[seed]["summary"]
        off_summary = attention_off[seed]["summary"]

        accuracy_delta = (
            on_summary["test_accuracy"]
            - off_summary["test_accuracy"]
        )

        f1_delta = (
            on_summary["test_f1_score"]
            - off_summary["test_f1_score"]
        )

        paired_results.append(
            {
                "seed": seed,
                "accuracy_attention_on":
                    on_summary["test_accuracy"],
                "accuracy_attention_off":
                    off_summary["test_accuracy"],
                "accuracy_delta": accuracy_delta,
                "f1_attention_on":
                    on_summary["test_f1_score"],
                "f1_attention_off":
                    off_summary["test_f1_score"],
                "f1_delta": f1_delta,
            }
        )

    accuracy_deltas = [
        row["accuracy_delta"]
        for row in paired_results
    ]

    f1_deltas = [
        row["f1_delta"]
        for row in paired_results
    ]

    # --------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------

    aggregate_summary = {
        "experiment": "EXP-004",
        "description":
            "Multi-seed paired attention ablation",
        "seeds": SEEDS,
        "number_of_seeds": len(SEEDS),
        "number_of_training_runs": len(results),
        "attention_on": {
            "accuracy": calculate_statistics(on_accuracy),
            "f1_score": calculate_statistics(on_f1),
        },
        "attention_off": {
            "accuracy": calculate_statistics(off_accuracy),
            "f1_score": calculate_statistics(off_f1),
        },
        "paired_delta_on_minus_off": {
            "accuracy":
                calculate_statistics(accuracy_deltas),
            "f1_score":
                calculate_statistics(f1_deltas),
        },
        "paired_results": paired_results,
    }

    # --------------------------------------------------
    # Console summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("EXP-004 Multi-seed Attention Ablation Summary")
    print("=" * 60)

    print()
    print("Attention ON")
    print(
        "Accuracy:",
        f"mean={mean(on_accuracy):.4f}, "
        f"std={stdev(on_accuracy):.4f}",
    )
    print(
        "F1-score:",
        f"mean={mean(on_f1):.4f}, "
        f"std={stdev(on_f1):.4f}",
    )

    print()
    print("Attention OFF")
    print(
        "Accuracy:",
        f"mean={mean(off_accuracy):.4f}, "
        f"std={stdev(off_accuracy):.4f}",
    )
    print(
        "F1-score:",
        f"mean={mean(off_f1):.4f}, "
        f"std={stdev(off_f1):.4f}",
    )

    print()
    print("-" * 60)
    print("Paired results (Attention ON - OFF)")
    print("-" * 60)

    for row in paired_results:
        print(
            f"Seed {row['seed']:>3}: "
            f"Accuracy delta={row['accuracy_delta']:+.4f}, "
            f"F1 delta={row['f1_delta']:+.4f}"
        )

    print()
    print(
        "Mean accuracy delta:",
        f"{mean(accuracy_deltas):+.4f}",
        f"(std={stdev(accuracy_deltas):.4f})",
    )

    print(
        "Mean F1 delta      :",
        f"{mean(f1_deltas):+.4f}",
        f"(std={stdev(f1_deltas):.4f})",
    )

    output_dir = save_results(
        results=results,
        paired_results=paired_results,
        aggregate_summary=aggregate_summary,
    )

    print()
    print("EXP-004 aggregate results saved to:")
    print(output_dir)
    print()
    print("Files:")
    print("  exp004_runs.csv")
    print("  exp004_paired_deltas.csv")
    print("  exp004_summary.json")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
