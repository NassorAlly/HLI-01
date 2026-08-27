"""
run_temporal_masking_robustness.py

EXP-009: Temporal frame masking robustness evaluation
for HLI-01 v1.0.0.

Evaluates the frozen BiLSTM + Attention classifier when
complete landmark frames within the 30-frame input sequence
are synthetically masked.

Important:
This experiment measures classifier sensitivity to missing
temporal information. It does NOT simulate the production
real-time buffer recovery policy.
"""

import torch

from run_input_robustness import (
    REFERENCE_CHECKPOINT,
    REFERENCE_SEED,
    build_model,
    get_test_loader,
)

from src.config.settings import SEQUENCE_LENGTH
from src.evaluation.evaluator import Evaluator


MASK_FRACTIONS = [
    0.00,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
]

MASK_SEEDS = [
    9001,
    9002,
    9003,
    9004,
    9005,
]


def apply_temporal_mask(
    inputs,
    mask_fraction,
    generator,
):
    """
    Mask an exact number of complete temporal frames
    independently for each sample in a batch.
    """

    if mask_fraction == 0.0:
        return inputs.clone()

    masked_inputs = inputs.clone()

    batch_size = masked_inputs.shape[0]
    sequence_length = masked_inputs.shape[1]

    frames_to_mask = round(
        sequence_length * mask_fraction
    )

    for sample_index in range(batch_size):

        frame_indices = torch.randperm(
            sequence_length,
            generator=generator,
            device=inputs.device,
        )[:frames_to_mask]

        masked_inputs[
            sample_index,
            frame_indices,
            :,
        ] = 0.0

    return masked_inputs


def evaluate_once(
    model,
    test_loader,
    device,
    mask_fraction,
    mask_seed,
):
    y_true = []
    y_pred = []

    generator = torch.Generator(
        device=device.type
    )
    generator.manual_seed(mask_seed)

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            masked_inputs = apply_temporal_mask(
                inputs=inputs,
                mask_fraction=mask_fraction,
                generator=generator,
            )

            outputs = model(masked_inputs)

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            y_true.extend(
                labels.cpu().tolist()
            )
            y_pred.extend(
                predictions.cpu().tolist()
            )

    evaluation = Evaluator.evaluate(
        y_true,
        y_pred,
    )

    return {
        "accuracy":
            float(evaluation["accuracy"]),
        "precision":
            float(evaluation["precision"]),
        "recall":
            float(evaluation["recall"]),
        "f1_score":
            float(evaluation["f1_score"]),
        "confusion_matrix":
            evaluation[
                "confusion_matrix"
            ].tolist(),
    }




# --------------------------------------------------
# EXP-009 aggregate experiment runner
# --------------------------------------------------

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


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


def save_results(
    run_results,
    aggregate_results,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = (
        OUTPUT_ROOT
        / (
            f"EXP_{timestamp}_"
            "exp009_temporal_masking_robustness"
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_csv = (
        output_dir
        / "exp009_masking_runs.csv"
    )

    with run_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "mask_fraction",
                "frames_masked",
                "mask_seed",
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
                    "mask_fraction":
                        row["mask_fraction"],
                    "frames_masked":
                        row["frames_masked"],
                    "mask_seed":
                        row["mask_seed"],
                    "accuracy":
                        row["accuracy"],
                    "precision":
                        row["precision"],
                    "recall":
                        row["recall"],
                    "f1_score":
                        row["f1_score"],
                }
            )

    summary_json = (
        output_dir
        / "exp009_masking_summary.json"
    )

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


def run_experiment():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print(
        "EXP-009 Temporal Frame "
        "Masking Robustness"
    )
    print("=" * 60)

    print()
    print("Device              :", device)
    print(
        "Reference checkpoint:",
        REFERENCE_CHECKPOINT,
    )
    print(
        "Reference seed      :",
        REFERENCE_SEED,
    )
    print(
        "Sequence length     :",
        SEQUENCE_LENGTH,
    )

    model = build_model(device)
    test_loader = get_test_loader()

    run_results = []

    for mask_fraction in MASK_FRACTIONS:

        frames_masked = round(
            SEQUENCE_LENGTH
            * mask_fraction
        )

        seeds = (
            [0]
            if mask_fraction == 0.0
            else MASK_SEEDS
        )

        print()
        print("-" * 60)
        print(
            "Mask fraction:",
            f"{mask_fraction:.2f}",
        )
        print(
            "Frames masked:",
            frames_masked,
        )
        print("-" * 60)

        for mask_seed in seeds:

            result = evaluate_once(
                model=model,
                test_loader=test_loader,
                device=device,
                mask_fraction=
                    mask_fraction,
                mask_seed=
                    mask_seed,
            )

            row = {
                "mask_fraction":
                    mask_fraction,
                "frames_masked":
                    frames_masked,
                "mask_seed":
                    mask_seed,
                **result,
            }

            run_results.append(row)

            print(
                f"seed={mask_seed:>4} "
                f"accuracy="
                f"{result['accuracy']:.4f} "
                f"f1="
                f"{result['f1_score']:.4f}"
            )

    clean_result = next(
        row
        for row in run_results
        if row["mask_fraction"] == 0.0
    )

    clean_accuracy = (
        clean_result["accuracy"]
    )
    clean_f1 = (
        clean_result["f1_score"]
    )

    levels = []

    for mask_fraction in MASK_FRACTIONS:

        subset = [
            row
            for row in run_results
            if row["mask_fraction"]
            == mask_fraction
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

        accuracy_stats = (
            calculate_statistics(
                accuracy_values
            )
        )

        f1_stats = (
            calculate_statistics(
                f1_values
            )
        )

        levels.append(
            {
                "mask_fraction":
                    mask_fraction,
                "frames_masked":
                    subset[0][
                        "frames_masked"
                    ],
                "evaluations":
                    len(subset),
                "accuracy":
                    accuracy_stats,
                "precision":
                    calculate_statistics(
                        precision_values
                    ),
                "recall":
                    calculate_statistics(
                        recall_values
                    ),
                "f1_score":
                    f1_stats,
                "accuracy_drop_from_clean":
                    clean_accuracy
                    - accuracy_stats["mean"],
                "f1_drop_from_clean":
                    clean_f1
                    - f1_stats["mean"],
            }
        )

    aggregate_results = {
        "experiment":
            "EXP-009",
        "description":
            (
                "Temporal frame masking "
                "robustness evaluation"
            ),
        "reference_model": {
            "architecture":
                "BiLSTM_Attention",
            "training_seed":
                REFERENCE_SEED,
            "checkpoint":
                str(REFERENCE_CHECKPOINT),
        },
        "sequence_length":
            SEQUENCE_LENGTH,
        "mask_fractions":
            MASK_FRACTIONS,
        "mask_seeds":
            MASK_SEEDS,
        "number_of_evaluations":
            len(run_results),
        "levels":
            levels,
    }

    print()
    print("=" * 60)
    print("EXP-009 Masking Summary")
    print("=" * 60)

    for level in levels:

        print()
        print(
            "Mask:",
            f"{level['mask_fraction'] * 100:.0f}% "
            f"({level['frames_masked']} frames)"
        )

        print(
            "Accuracy:",
            f"mean="
            f"{level['accuracy']['mean']:.4f}, "
            f"std="
            f"{level['accuracy']['std']:.4f}"
        )

        print(
            "F1-score:",
            f"mean="
            f"{level['f1_score']['mean']:.4f}, "
            f"std="
            f"{level['f1_score']['std']:.4f}"
        )

        print(
            "Accuracy drop:",
            f"{level['accuracy_drop_from_clean']:.4f}"
        )

        print(
            "F1 drop      :",
            f"{level['f1_drop_from_clean']:.4f}"
        )

    output_dir = save_results(
        run_results=run_results,
        aggregate_results=
            aggregate_results,
    )

    print()
    print("EXP-009 results saved to:")
    print(output_dir)

    print()
    print("Files:")
    print(
        "  exp009_masking_runs.csv"
    )
    print(
        "  exp009_masking_summary.json"
    )
    print("=" * 60)

    return aggregate_results


if __name__ == "__main__":
    run_experiment()
