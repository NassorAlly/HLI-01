"""
run_multiseed_experiment.py

EXP-002: Multi-seed validation runner for HLI-01 v1.0.0.
"""

from statistics import mean, stdev

from run_experiment import run_experiment


SEEDS = [42, 7, 21, 84, 123]


def main():
    results = []

    for seed in SEEDS:
        experiment_dir, _, _, summary = run_experiment(
            experiment_name=f"multiseed_seed_{seed}",
            seed=seed,
        )

        results.append(
            {
                "seed": seed,
                "experiment_dir": str(experiment_dir),
                "summary": summary,
            }
        )

    metrics = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
    ]

    print()
    print("=" * 60)
    print("EXP-002 Multi-seed Summary")
    print("=" * 60)

    for metric in metrics:
        values = [
            result["summary"][metric]
            for result in results
        ]

        print(
            f"{metric}: "
            f"mean={mean(values):.4f}, "
            f"std={stdev(values):.4f}"
        )

    return results


if __name__ == "__main__":
    main()
