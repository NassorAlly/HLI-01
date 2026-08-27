"""
run_buffer_recovery_ablation.py

EXP-008: Temporal buffer recovery strategy ablation
for HLI-01 v1.0.0.

Compares alternative temporal buffering policies under
intermittent landmark-detection failures.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

import run_detection_interruption_robustness as exp007


POLICIES = {
    "full_reset": 0,
    "one_miss_tolerance": 1,
    "two_miss_tolerance": 2,
    "skip_miss": None,
}

OUTPUT_ROOT = Path("outputs/experiments")
BASE_RANDOM_SEED = 8000


def simulate_trial(
    detection_probability,
    rng,
    policy,
):
    valid_frames = 0
    consecutive_misses = 0
    resets = 0
    total_misses = 0

    miss_tolerance = POLICIES[policy]

    for camera_frame in range(
        1,
        exp007.MAX_CAMERA_FRAMES + 1,
    ):
        detected = (
            rng.random()
            < detection_probability
        )

        if detected:
            valid_frames += 1
            consecutive_misses = 0

            if (
                valid_frames
                == exp007.SEQUENCE_LENGTH
            ):
                return {
                    "success": True,
                    "frames_required":
                        camera_frame,
                    "resets":
                        resets,
                    "misses":
                        total_misses,
                }

        else:
            total_misses += 1
            consecutive_misses += 1

            if policy == "skip_miss":
                continue

            if (
                consecutive_misses
                > miss_tolerance
            ):
                if valid_frames > 0:
                    resets += 1

                valid_frames = 0
                consecutive_misses = 0

    return {
        "success": False,
        "frames_required": None,
        "resets": resets,
        "misses": total_misses,
    }


def calculate_statistics(values):
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "p95": None,
        }

    return {
        "mean": mean(values),
        "median": median(values),
        "std": (
            stdev(values)
            if len(values) > 1
            else 0.0
        ),
        "p95": float(
            np.percentile(values, 95)
        ),
    }


def summarize_condition(
    policy,
    detection_probability,
    results,
):
    successful = [
        result
        for result in results
        if result["success"]
    ]

    frames = [
        result["frames_required"]
        for result in successful
    ]

    resets = [
        result["resets"]
        for result in results
    ]

    misses = [
        result["misses"]
        for result in results
    ]

    frame_stats = (
        calculate_statistics(frames)
    )

    return {
        "policy": policy,
        "detection_probability":
            detection_probability,
        "failure_probability":
            1.0 - detection_probability,
        "trials": len(results),
        "successes": len(successful),
        "success_rate":
            len(successful) / len(results),
        "mean_frames_required":
            frame_stats["mean"],
        "median_frames_required":
            frame_stats["median"],
        "std_frames_required":
            frame_stats["std"],
        "p95_frames_required":
            frame_stats["p95"],
        "mean_resets":
            mean(resets),
        "mean_misses":
            mean(misses),
    }


def save_results(
    trial_rows,
    summaries,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        OUTPUT_ROOT
        / (
            f"EXP_{timestamp}_"
            "exp008_buffer_recovery_ablation"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trials_csv = (
        output_dir
        / "exp008_trials.csv"
    )

    with trials_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "policy",
                "detection_probability",
                "trial",
                "success",
                "frames_required",
                "resets",
                "misses",
            ],
        )

        writer.writeheader()
        writer.writerows(trial_rows)

    summary_csv = (
        output_dir
        / "exp008_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "policy",
                "detection_probability",
                "failure_probability",
                "trials",
                "successes",
                "success_rate",
                "mean_frames_required",
                "median_frames_required",
                "std_frames_required",
                "p95_frames_required",
                "mean_resets",
                "mean_misses",
            ],
        )

        writer.writeheader()
        writer.writerows(summaries)

    summary_json = (
        output_dir
        / "exp008_summary.json"
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                "experiment": "EXP-008",
                "description":
                    (
                        "Temporal buffer recovery "
                        "strategy ablation"
                    ),
                "policies": POLICIES,
                "sequence_length":
                    exp007.SEQUENCE_LENGTH,
                "detection_probabilities":
                    exp007.DETECTION_PROBABILITIES,
                "trials_per_condition":
                    exp007.NUM_TRIALS,
                "max_camera_frames":
                    exp007.MAX_CAMERA_FRAMES,
                "conditions":
                    summaries,
            },
            file,
            indent=4,
        )

    return output_dir


def main():
    print("=" * 60)
    print(
        "EXP-008 Temporal Buffer "
        "Recovery Strategy Ablation"
    )
    print("=" * 60)

    print()
    print(
        "Sequence length       :",
        exp007.SEQUENCE_LENGTH,
    )
    print(
        "Trials per condition  :",
        exp007.NUM_TRIALS,
    )
    print(
        "Maximum camera frames :",
        exp007.MAX_CAMERA_FRAMES,
    )
    print(
        "Policies              :",
        list(POLICIES.keys()),
    )

    trial_rows = []
    summaries = []

    for policy in POLICIES:
        for level_index, probability in enumerate(
            exp007.DETECTION_PROBABILITIES
        ):
            results = []

            for trial in range(
                1,
                exp007.NUM_TRIALS + 1,
            ):
                rng = np.random.default_rng(
                    BASE_RANDOM_SEED
                    + level_index * 10000
                    + trial
                )

                result = simulate_trial(
                    detection_probability=
                        probability,
                    rng=rng,
                    policy=policy,
                )

                results.append(result)

                trial_rows.append(
                    {
                        "policy": policy,
                        "detection_probability":
                            probability,
                        "trial": trial,
                        "success":
                            result["success"],
                        "frames_required":
                            result[
                                "frames_required"
                            ],
                        "resets":
                            result["resets"],
                        "misses":
                            result["misses"],
                    }
                )

            summary = summarize_condition(
                policy=policy,
                detection_probability=
                    probability,
                results=results,
            )

            summaries.append(summary)

            print()
            print("-" * 60)
            print("Policy               :", policy)
            print(
                "Detection probability:",
                f"{probability:.0%}",
            )
            print(
                "Success rate          :",
                f"{summary['success_rate']:.2%}",
            )
            print(
                "Mean frames required  :",
                summary[
                    "mean_frames_required"
                ],
            )
            print(
                "95th percentile       :",
                summary[
                    "p95_frames_required"
                ],
            )
            print(
                "Mean resets           :",
                f"{summary['mean_resets']:.2f}",
            )

    output_dir = save_results(
        trial_rows=trial_rows,
        summaries=summaries,
    )

    print()
    print("=" * 60)
    print("EXP-008 Summary")
    print("=" * 60)

    for probability in (
        exp007.DETECTION_PROBABILITIES
    ):
        print()
        print(
            "Detection probability:",
            f"{probability:.0%}",
        )

        subset = [
            summary
            for summary in summaries
            if (
                summary[
                    "detection_probability"
                ]
                == probability
            )
        ]

        for summary in subset:
            print(
                f"  "
                f"{summary['policy']:<20} "
                f"success="
                f"{summary['success_rate']:.2%}, "
                f"mean_frames="
                f"{summary['mean_frames_required']}"
            )

    print()
    print("EXP-008 results saved to:")
    print(output_dir)

    print()
    print("Files:")
    print("  exp008_trials.csv")
    print("  exp008_summary.csv")
    print("  exp008_summary.json")
    print("=" * 60)

    return summaries


if __name__ == "__main__":
    main()
