"""
run_input_robustness.py

EXP-005: Input perturbation robustness evaluation
for HLI-01 v1.0.0.

Evaluates a frozen BiLSTM + Attention checkpoint under
controlled Gaussian perturbations of the test inputs.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import torch

from src.config.settings import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_CLASSES,
)
from src.models.lstm_model import LSTMModel
from src.training.data_loader import DataLoaderManager
from src.training.checkpoint_manager import CheckpointManager
from src.evaluation.evaluator import Evaluator


REFERENCE_CHECKPOINT = Path(
    "outputs/experiments/"
    "EXP_20260827_213541_exp004_seed_7_attention_on/"
    "checkpoints/best_model.pth"
)

OUTPUT_ROOT = Path("outputs/experiments")

REFERENCE_SEED = 7

NOISE_FRACTIONS = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.40,
]

NOISE_SEEDS = [
    1001,
    1002,
    1003,
    1004,
    1005,
]


def build_model(device):
    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        use_attention=True,
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=REFERENCE_CHECKPOINT.parent
    )

    checkpoint_manager.load_model(
        model=model,
        filename=REFERENCE_CHECKPOINT.name,
        device=device,
    )

    model = model.to(device)
    model.eval()

    return model


def get_test_loader():
    data_manager = DataLoaderManager()
    _, _, test_loader = data_manager.create()

    return test_loader


def calculate_input_std(test_loader):
    batches = []

    for inputs, _ in test_loader:
        batches.append(inputs)

    all_inputs = torch.cat(
        batches,
        dim=0,
    )

    return float(all_inputs.std())


def apply_gaussian_noise(
    inputs,
    noise_sigma,
    generator,
):
    if noise_sigma == 0.0:
        return inputs

    noise = torch.randn(
        inputs.shape,
        dtype=inputs.dtype,
        device=inputs.device,
        generator=generator,
    )

    return inputs + noise_sigma * noise


def evaluate_once(
    model,
    test_loader,
    device,
    noise_sigma,
    noise_seed,
):
    y_true = []
    y_pred = []

    generator = torch.Generator(
        device=device.type
    )

    generator.manual_seed(noise_seed)

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            perturbed_inputs = apply_gaussian_noise(
                inputs=inputs,
                noise_sigma=noise_sigma,
                generator=generator,
            )

            outputs = model(perturbed_inputs)

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
            evaluation["confusion_matrix"].tolist(),
    }


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
        / f"EXP_{timestamp}_exp005_input_robustness"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_csv = (
        output_dir
        / "exp005_robustness_runs.csv"
    )

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
                    "noise_fraction":
                        row["noise_fraction"],
                    "noise_sigma":
                        row["noise_sigma"],
                    "noise_seed":
                        row["noise_seed"],
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
        / "exp005_robustness_summary.json"
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


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("EXP-005 Input Perturbation Robustness")
    print("=" * 60)

    print()
    print("Device              :", device)
    print(
        "Reference checkpoint:",
        REFERENCE_CHECKPOINT,
    )

    model = build_model(device)
    test_loader = get_test_loader()

    input_std = calculate_input_std(
        test_loader
    )

    print(
        "Empirical input std :",
        f"{input_std:.6f}",
    )

    run_results = []

    # --------------------------------------------------
    # Evaluate clean and noisy conditions
    # --------------------------------------------------

    for noise_fraction in NOISE_FRACTIONS:

        noise_sigma = (
            noise_fraction
            * input_std
        )

        if noise_fraction == 0.0:
            seeds = [0]
        else:
            seeds = NOISE_SEEDS

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

            result = evaluate_once(
                model=model,
                test_loader=test_loader,
                device=device,
                noise_sigma=noise_sigma,
                noise_seed=noise_seed,
            )

            row = {
                "noise_fraction":
                    noise_fraction,
                "noise_sigma":
                    noise_sigma,
                "noise_seed":
                    noise_seed,
                **result,
            }

            run_results.append(row)

            print(
                f"seed={noise_seed:>4} "
                f"accuracy={result['accuracy']:.4f} "
                f"f1={result['f1_score']:.4f}"
            )

    # --------------------------------------------------
    # Aggregate results
    # --------------------------------------------------

    aggregate_levels = []

    clean_accuracy = None
    clean_f1 = None

    for noise_fraction in NOISE_FRACTIONS:

        subset = [
            row
            for row in run_results
            if row["noise_fraction"]
            == noise_fraction
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

        precision_stats = (
            calculate_statistics(
                precision_values
            )
        )

        recall_stats = (
            calculate_statistics(
                recall_values
            )
        )

        f1_stats = (
            calculate_statistics(
                f1_values
            )
        )

        if noise_fraction == 0.0:
            clean_accuracy = (
                accuracy_stats["mean"]
            )
            clean_f1 = (
                f1_stats["mean"]
            )

        aggregate_levels.append(
            {
                "noise_fraction":
                    noise_fraction,
                "noise_sigma":
                    subset[0]["noise_sigma"],
                "evaluations":
                    len(subset),
                "accuracy":
                    accuracy_stats,
                "precision":
                    precision_stats,
                "recall":
                    recall_stats,
                "f1_score":
                    f1_stats,
            }
        )

    for level in aggregate_levels:
        level["accuracy_drop_from_clean"] = (
            clean_accuracy
            - level["accuracy"]["mean"]
        )

        level["f1_drop_from_clean"] = (
            clean_f1
            - level["f1_score"]["mean"]
        )

    aggregate_results = {
        "experiment": "EXP-005",
        "description":
            "Gaussian input perturbation robustness",
        "reference_model": {
            "architecture":
                "BiLSTM_Attention",
            "training_seed":
                REFERENCE_SEED,
            "checkpoint":
                str(REFERENCE_CHECKPOINT),
        },
        "input_standard_deviation":
            input_std,
        "noise_fractions":
            NOISE_FRACTIONS,
        "noise_seeds":
            NOISE_SEEDS,
        "number_of_evaluations":
            len(run_results),
        "levels":
            aggregate_levels,
    }

    print()
    print("=" * 60)
    print("EXP-005 Robustness Summary")
    print("=" * 60)

    for level in aggregate_levels:

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

    output_dir = save_results(
        run_results=run_results,
        aggregate_results=aggregate_results,
    )

    print()
    print(
        "EXP-005 results saved to:"
    )
    print(output_dir)

    print()
    print("Files:")
    print(
        "  exp005_robustness_runs.csv"
    )
    print(
        "  exp005_robustness_summary.json"
    )

    print("=" * 60)

    return aggregate_results


if __name__ == "__main__":
    main()
