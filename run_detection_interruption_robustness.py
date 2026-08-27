"""
run_detection_interruption_robustness.py

EXP-007: Landmark detection interruption robustness
for HLI-01 v1.0.0.

Simulates intermittent hand-landmark detection failures
under the current real-time inference policy, where a
missed detection clears the accumulated sequence.

Primary outcome:
Number of camera frames required to obtain
SEQUENCE_LENGTH consecutive valid detections.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

from src.config.settings import SEQUENCE_LENGTH


DETECTION_PROBABILITIES = [
    1.00,
    0.99,
    0.95,
    0.90,
    0.80,
    0.70,
]

NUM_TRIALS = 1000
MAX_CAMERA_FRAMES = 300
BASE_RANDOM_SEED = 7000

OUTPUT_ROOT = Path("outputs/experiments")


def simulate_trial(
    detection_probability,
    rng,
):
    consecutive_valid = 0
    resets = 0

    for camera_frame in range(
        1,
        MAX_CAMERA_FRAMES + 1,
    ):
        detected = (
            rng.random()
            < detection_probability
        )

        if detected:
            consecutive_valid += 1

            if (
                consecutive_valid
                == SEQUENCE_LENGTH
            ):
                return {
                    "success": True,
                    "frames_required":
                        camera_frame,
                    "resets":
                        resets,
                }

        else:
            if consecutive_valid > 0:
                resets += 1

            consecutive_valid = 0

    return {
        "success": False,
        "frames_required": None,
        "resets": resets,
    }


def percentile(values, q):
    return float(
        np.percentile(
            values,
            q,
        )
    )


def summarize_level(
    detection_probability,
    trial_results,
):
    successful = [
        result
        for result in trial_results
        if result["success"]
    ]

    success_rate = (
        len(successful)
        / len(trial_results)
    )

    frames_required = [
        result["frames_required"]
        for result in successful
    ]

    resets = [
        result["resets"]
        for result in trial_results
    ]

    if frames_required:
        frame_mean = mean(
            frames_required
        )

        frame_median = median(
            frames_required
        )

        frame_std = (
            stdev(frames_required)
            if len(frames_required) > 1
            else 0.0
        )

        frame_p95 = percentile(
            frames_required,
            95,
        )
    else:
        frame_mean = None
        frame_median = None
        frame_std = None
        frame_p95 = None

    return {
        "detection_probability":
            detection_probability,
        "detection_failure_probability":
            1.0
            - detection_probability,
        "trials":
            len(trial_results),
        "successes":
            len(successful),
        "success_rate":
            success_rate,
        "mean_frames_required":
            frame_mean,
        "median_frames_required":
            frame_median,
        "std_frames_required":
            frame_std,
        "p95_frames_required":
            frame_p95,
        "mean_resets":
            mean(resets),
        "max_camera_frames":
            MAX_CAMERA_FRAMES,
        "required_consecutive_frames":
            SEQUENCE_LENGTH,
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
            "exp007_detection_interruption"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trial_csv = (
        output_dir
        / "exp007_trials.csv"
    )

    with trial_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "detection_probability",
                "trial",
                "success",
                "frames_required",
                "resets",
            ],
        )

        writer.writeheader()
        writer.writerows(
            trial_rows
        )

    summary_csv = (
        output_dir
        / "exp007_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "detection_probability",
                "detection_failure_probability",
                "trials",
                "successes",
                "success_rate",
                "mean_frames_required",
                "median_frames_required",
                "std_frames_required",
                "p95_frames_required",
                "mean_resets",
                "max_camera_frames",
                "required_consecutive_frames",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )

    summary_json = (
        output_dir
        / "exp007_summary.json"
    )

    with summary_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "experiment":
                    "EXP-007",
                "description":
                    (
                        "Landmark detection "
                        "interruption robustness"
                    ),
                "policy":
                    (
                        "A missed detection clears "
                        "the accumulated sequence."
                    ),
                "required_consecutive_frames":
                    SEQUENCE_LENGTH,
                "max_camera_frames":
                    MAX_CAMERA_FRAMES,
                "trials_per_level":
                    NUM_TRIALS,
                "detection_probabilities":
                    DETECTION_PROBABILITIES,
                "levels":
                    summaries,
            },
            file,
            indent=4,
        )

    return output_dir


def main():
    print("=" * 60)
    print(
        "EXP-007 Landmark Detection "
        "Interruption Robustness"
    )
    print("=" * 60)

    print()
    print(
        "Required consecutive frames:",
        SEQUENCE_LENGTH,
    )
    print(
        "Trials per level          :",
        NUM_TRIALS,
    )
    print(
        "Maximum camera frames     :",
        MAX_CAMERA_FRAMES,
    )

    trial_rows = []
    summaries = []

    for level_index, probability in enumerate(
        DETECTION_PROBABILITIES
    ):
        rng = np.random.default_rng(
            BASE_RANDOM_SEED
            + level_index
        )

        trial_results = []

        for trial in range(
            1,
            NUM_TRIALS + 1,
        ):
            result = simulate_trial(
                detection_probability=
                    probability,
                rng=rng,
            )

            trial_results.append(
                result
            )

            trial_rows.append(
                {
                    "detection_probability":
                        probability,
                    "trial":
                        trial,
                    "success":
                        result["success"],
                    "frames_required":
                        result[
                            "frames_required"
                        ],
                    "resets":
                        result["resets"],
                }
            )

        summary = summarize_level(
            detection_probability=
                probability,
            trial_results=
                trial_results,
        )

        summaries.append(
            summary
        )

        print()
        print("-" * 60)
        print(
            "Detection probability:",
            f"{probability:.0%}",
        )
        print(
            "Failure probability  :",
            f"{1.0 - probability:.0%}",
        )
        print(
            "Success within window:",
            f"{summary['success_rate']:.2%}",
        )
        print(
            "Mean frames required :",
            summary[
                "mean_frames_required"
            ],
        )
        print(
            "Median frames        :",
            summary[
                "median_frames_required"
            ],
        )
        print(
            "95th percentile      :",
            summary[
                "p95_frames_required"
            ],
        )
        print(
            "Mean sequence resets :",
            f"{summary['mean_resets']:.2f}",
        )

    output_dir = save_results(
        trial_rows=trial_rows,
        summaries=summaries,
    )

    print()
    print("=" * 60)
    print("EXP-007 Summary")
    print("=" * 60)

    for summary in summaries:
        print(
            f"Detection "
            f"{summary['detection_probability']:.0%}: "
            f"success="
            f"{summary['success_rate']:.2%}, "
            f"mean_frames="
            f"{summary['mean_frames_required']}, "
            f"p95="
            f"{summary['p95_frames_required']}"
        )

    print()
    print("EXP-007 results saved to:")
    print(output_dir)

    print()
    print("Files:")
    print("  exp007_trials.csv")
    print("  exp007_summary.csv")
    print("  exp007_summary.json")
    print("=" * 60)

    return summaries


if __name__ == "__main__":
    main()
