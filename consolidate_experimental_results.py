"""
consolidate_experimental_results.py

HLI-01 v1.0.0
Experimental Results Consolidation

Consolidates the validated results from EXP-001
through EXP-010 into a single research-ready
results package.

The script reads existing experiment outputs and
does not rerun model training or evaluation.
"""

import csv
import json
from datetime import datetime
from pathlib import Path


EXPERIMENT_ROOT = Path("outputs/experiments")
OUTPUT_ROOT = Path("outputs/research_summary")


def load_json(path):
    """
    Load a JSON file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Required result file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    print("=" * 60)
    print("HLI-01 Experimental Results Consolidation")
    print("=" * 60)

    print()
    print("Experiment root :", EXPERIMENT_ROOT)
    print("Output root     :", OUTPUT_ROOT)

    print()
    print(
        "Consolidation module initialized successfully."
    )


if __name__ == "__main__":
    main()


AUTHORITATIVE_RESULTS = {
    "EXP-001": (
        EXPERIMENT_ROOT
        / "EXP_20260819_234132_baseline"
        / "metrics.json"
    ),
    "EXP-004": (
        EXPERIMENT_ROOT
        / "EXP_20260827_213730_exp004_attention_ablation_summary"
        / "exp004_summary.json"
    ),
    "EXP-005": (
        EXPERIMENT_ROOT
        / "EXP_20260827_220940_exp005_input_robustness"
        / "exp005_robustness_summary.json"
    ),
    "EXP-006": (
        EXPERIMENT_ROOT
        / "EXP_20260827_223417_exp006_robustness_threshold"
        / "exp006_threshold_summary.json"
    ),
    "EXP-007": (
        EXPERIMENT_ROOT
        / "EXP_20260827_224733_exp007_detection_interruption"
        / "exp007_summary.json"
    ),
    "EXP-008": (
        EXPERIMENT_ROOT
        / "EXP_20260827_230254_exp008_buffer_recovery_ablation"
        / "exp008_summary.json"
    ),
    "EXP-009": (
        EXPERIMENT_ROOT
        / "EXP_20260827_232045_exp009_temporal_masking_robustness"
        / "exp009_masking_summary.json"
    ),
    "EXP-010": (
        EXPERIMENT_ROOT
        / "EXP_20260827_233944_exp010_inference_efficiency"
        / "exp010_efficiency_summary.json"
    ),
}


EXP002_METRICS = sorted(
    EXPERIMENT_ROOT.glob(
        "EXP_20260820_*_multiseed_seed_*/metrics.json"
    )
)


EXP003_RESULTS = {
    "attention_on": (
        EXPERIMENT_ROOT
        / "EXP_20260820_010936_exp003_attention_on"
        / "metrics.json"
    ),
    "attention_off": (
        EXPERIMENT_ROOT
        / "EXP_20260820_010941_exp003_attention_off"
        / "metrics.json"
    ),
}


def validate_result_files():
    """
    Confirm that every authoritative result file exists.
    """

    missing = []

    for experiment, path in AUTHORITATIVE_RESULTS.items():
        if not path.exists():
            missing.append(
                f"{experiment}: {path}"
            )

    for variant, path in EXP003_RESULTS.items():
        if not path.exists():
            missing.append(
                f"EXP-003 {variant}: {path}"
            )

    if len(EXP002_METRICS) != 5:
        missing.append(
            "EXP-002 expected exactly 5 metrics files, "
            f"found {len(EXP002_METRICS)}"
        )

    if missing:
        raise FileNotFoundError(
            "Authoritative experiment results are incomplete:\n"
            + "\n".join(missing)
        )

    return True


def consolidate_exp001():
    """
    Consolidate EXP-001 baseline results.
    """

    data = load_json(
        AUTHORITATIVE_RESULTS["EXP-001"]
    )

    return {
        "experiment": "EXP-001",
        "title": "Baseline Validation",
        "category": "baseline",
        "test_accuracy":
            data["test_accuracy"],
        "test_precision":
            data["test_precision"],
        "test_recall":
            data["test_recall"],
        "test_f1_score":
            data["test_f1_score"],
        "best_epoch":
            data["best_epoch"],
        "best_validation_loss":
            data["best_validation_loss"],
        "epochs_completed":
            data["epochs_completed"],
        "device":
            data["device"],
    }


def consolidate_exp002():
    """
    Consolidate EXP-002 multi-seed validation.
    """

    from statistics import mean, stdev

    runs = []

    for path in EXP002_METRICS:
        data = load_json(path)

        runs.append(
            {
                "seed":
                    data["runtime_config"]["seed"],
                "accuracy":
                    data["test_accuracy"],
                "precision":
                    data["test_precision"],
                "recall":
                    data["test_recall"],
                "f1_score":
                    data["test_f1_score"],
            }
        )

    accuracy = [
        run["accuracy"]
        for run in runs
    ]

    precision = [
        run["precision"]
        for run in runs
    ]

    recall = [
        run["recall"]
        for run in runs
    ]

    f1_score = [
        run["f1_score"]
        for run in runs
    ]

    return {
        "experiment": "EXP-002",
        "title": "Multi-seed Validation",
        "category": "reproducibility",
        "number_of_seeds":
            len(runs),
        "runs":
            runs,
        "accuracy": {
            "mean": mean(accuracy),
            "std": stdev(accuracy),
            "min": min(accuracy),
            "max": max(accuracy),
        },
        "precision": {
            "mean": mean(precision),
            "std": stdev(precision),
            "min": min(precision),
            "max": max(precision),
        },
        "recall": {
            "mean": mean(recall),
            "std": stdev(recall),
            "min": min(recall),
            "max": max(recall),
        },
        "f1_score": {
            "mean": mean(f1_score),
            "std": stdev(f1_score),
            "min": min(f1_score),
            "max": max(f1_score),
        },
    }


def consolidate_exp003():
    """
    Consolidate EXP-003 single-seed attention ablation.
    """

    attention_on = load_json(
        EXP003_RESULTS["attention_on"]
    )

    attention_off = load_json(
        EXP003_RESULTS["attention_off"]
    )

    return {
        "experiment": "EXP-003",
        "title": "Single-seed Attention Ablation",
        "category": "architecture_ablation",
        "seed":
            attention_on[
                "runtime_config"
            ]["seed"],
        "attention_on": {
            "accuracy":
                attention_on["test_accuracy"],
            "f1_score":
                attention_on["test_f1_score"],
            "best_validation_loss":
                attention_on[
                    "best_validation_loss"
                ],
        },
        "attention_off": {
            "accuracy":
                attention_off["test_accuracy"],
            "f1_score":
                attention_off["test_f1_score"],
            "best_validation_loss":
                attention_off[
                    "best_validation_loss"
                ],
        },
    }


def consolidate_exp004():
    """
    Consolidate EXP-004 multi-seed paired attention ablation.
    """

    data = load_json(
        AUTHORITATIVE_RESULTS["EXP-004"]
    )

    return {
        "experiment": "EXP-004",
        "title": "Multi-seed Paired Attention Ablation",
        "category": "architecture_ablation",
        "number_of_seeds":
            data["number_of_seeds"],
        "number_of_training_runs":
            data["number_of_training_runs"],
        "attention_on":
            data["attention_on"],
        "attention_off":
            data["attention_off"],
        "paired_delta_on_minus_off":
            data["paired_delta_on_minus_off"],
        "paired_results":
            data["paired_results"],
    }


def consolidate_exp005():
    """
    Consolidate EXP-005 Gaussian input robustness.
    """

    data = load_json(
        AUTHORITATIVE_RESULTS["EXP-005"]
    )

    return {
        "experiment": "EXP-005",
        "title": "Gaussian Input Perturbation Robustness",
        "category": "robustness",
        "reference_model":
            data["reference_model"],
        "input_standard_deviation":
            data["input_standard_deviation"],
        "number_of_evaluations":
            data["number_of_evaluations"],
        "levels":
            data["levels"],
    }


def consolidate_exp006():
    """
    Consolidate EXP-006 robustness failure threshold.
    """

    data = load_json(
        AUTHORITATIVE_RESULTS["EXP-006"]
    )

    return {
        "experiment": "EXP-006",
        "title": "Gaussian Robustness Failure Threshold",
        "category": "robustness",
        "reference_model":
            data["reference_model"],
        "input_standard_deviation":
            data["input_standard_deviation"],
        "number_of_evaluations":
            data["number_of_evaluations"],
        "first_observed_degradation":
            data["first_observed_degradation"],
        "levels":
            data["levels"],
    }


def consolidate_exp007():
    """
    Consolidate EXP-007 landmark detection interruption robustness.
    """

    data = load_json(
        AUTHORITATIVE_RESULTS["EXP-007"]
    )

    return {
        "experiment": "EXP-007",
        "title": "Landmark Detection Interruption Robustness",
        "category": "system_robustness",
        "policy":
            data["policy"],
        "required_consecutive_frames":
            data["required_consecutive_frames"],
        "max_camera_frames":
            data["max_camera_frames"],
        "trials_per_level":
            data["trials_per_level"],
        "levels":
            data["levels"],
    }


def consolidate_exp008():
    """
    Consolidate EXP-008 temporal buffer recovery ablation.
    """

    data = load_json(
        AUTHORITATIVE_RESULTS["EXP-008"]
    )

    return {
        "experiment": "EXP-008",
        "title": "Temporal Buffer Recovery Strategy Ablation",
        "category": "system_design_ablation",
        "policies":
            data["policies"],
        "sequence_length":
            data["sequence_length"],
        "trials_per_condition":
            data["trials_per_condition"],
        "max_camera_frames":
            data["max_camera_frames"],
        "conditions":
            data["conditions"],
    }


def consolidate_exp009():
    """
    Consolidate EXP-009 temporal masking robustness.
    """

    data = load_json(
        AUTHORITATIVE_RESULTS["EXP-009"]
    )

    return {
        "experiment": "EXP-009",
        "title": "Temporal Frame Masking Robustness",
        "category": "robustness",
        "reference_model":
            data["reference_model"],
        "sequence_length":
            data["sequence_length"],
        "number_of_evaluations":
            data["number_of_evaluations"],
        "levels":
            data["levels"],
    }


def consolidate_exp010():
    """
    Consolidate EXP-010 inference efficiency.
    """

    data = load_json(
        AUTHORITATIVE_RESULTS["EXP-010"]
    )

    return {
        "experiment": "EXP-010",
        "title": "Inference Performance and Computational Efficiency",
        "category": "efficiency",
        "reference_checkpoint":
            data["reference_checkpoint"],
        "checkpoint_size_mb":
            data["checkpoint_size_mb"],
        "parameters":
            data["parameters"],
        "warmup_iterations":
            data["warmup_iterations"],
        "timed_iterations":
            data["timed_iterations"],
        "batch_size":
            data["batch_size"],
        "input_shape":
            data["input_shape"],
        "benchmarks":
            data["benchmarks"],
    }


def build_master_summary():
    """
    Build the complete EXP-001 to EXP-010 master summary.
    """

    experiments = [
        consolidate_exp001(),
        consolidate_exp002(),
        consolidate_exp003(),
        consolidate_exp004(),
        consolidate_exp005(),
        consolidate_exp006(),
        consolidate_exp007(),
        consolidate_exp008(),
        consolidate_exp009(),
        consolidate_exp010(),
    ]

    return {
        "project": "HLI-01",
        "version": "v1.0.0",
        "description":
            (
                "Master experimental validation summary "
                "for EXP-001 through EXP-010"
            ),
        "number_of_experiments":
            len(experiments),
        "experiments":
            experiments,
    }


def build_master_table(master):
    """
    Create one concise research-facing row per experiment.
    """

    rows = []

    for exp in master["experiments"]:

        experiment = exp["experiment"]

        if experiment == "EXP-001":
            result = (
                f"Accuracy={exp['test_accuracy']:.4f}; "
                f"F1={exp['test_f1_score']:.4f}"
            )

        elif experiment == "EXP-002":
            result = (
                f"Accuracy="
                f"{exp['accuracy']['mean']:.4f} "
                f"± {exp['accuracy']['std']:.4f}; "
                f"F1="
                f"{exp['f1_score']['mean']:.4f} "
                f"± {exp['f1_score']['std']:.4f}"
            )

        elif experiment == "EXP-003":
            result = (
                f"Attention ON accuracy="
                f"{exp['attention_on']['accuracy']:.4f}; "
                f"OFF="
                f"{exp['attention_off']['accuracy']:.4f}"
            )

        elif experiment == "EXP-004":
            result = (
                f"Attention ON="
                f"{exp['attention_on']['accuracy']['mean']:.4f}; "
                f"OFF="
                f"{exp['attention_off']['accuracy']['mean']:.4f}; "
                f"paired delta="
                f"{exp['paired_delta_on_minus_off']['accuracy']['mean']:+.4f}"
            )

        elif experiment == "EXP-005":
            strongest = exp["levels"][-1]
            result = (
                f"Accuracy={strongest['accuracy']['mean']:.4f} "
                f"at noise fraction "
                f"{strongest['noise_fraction']:.2f}"
            )

        elif experiment == "EXP-006":
            first = exp[
                "first_observed_degradation"
            ]
            strongest = exp["levels"][-1]

            result = (
                f"First degradation at "
                f"{first['noise_fraction']:.2f}; "
                f"accuracy={strongest['accuracy']['mean']:.4f} "
                f"at {strongest['noise_fraction']:.2f}"
            )

        elif experiment == "EXP-007":
            level_90 = next(
                level
                for level in exp["levels"]
                if level[
                    "detection_probability"
                ] == 0.9
            )

            result = (
                f"Success={level_90['success_rate']:.3f} "
                f"at 90% detection"
            )

        elif experiment == "EXP-008":
            two_miss_70 = next(
                condition
                for condition in exp["conditions"]
                if (
                    condition["policy"]
                    == "two_miss_tolerance"
                    and condition[
                        "detection_probability"
                    ] == 0.7
                )
            )

            result = (
                f"Two-miss tolerance success="
                f"{two_miss_70['success_rate']:.3f} "
                f"at 70% detection"
            )

        elif experiment == "EXP-009":
            strongest = exp["levels"][-1]

            result = (
                f"Accuracy="
                f"{strongest['accuracy']['mean']:.4f} "
                f"with "
                f"{strongest['mask_fraction']:.0%} "
                f"frames masked"
            )

        elif experiment == "EXP-010":
            cpu = next(
                benchmark
                for benchmark in exp["benchmarks"]
                if benchmark["device"] == "cpu"
            )

            cuda = next(
                benchmark
                for benchmark in exp["benchmarks"]
                if benchmark["device"] == "cuda"
            )

            result = (
                f"Raw latency CPU="
                f"{cpu['raw_model']['mean_ms']:.4f} ms; "
                f"CUDA="
                f"{cuda['raw_model']['mean_ms']:.4f} ms"
            )

        else:
            result = ""

        rows.append(
            {
                "experiment":
                    experiment,
                "title":
                    exp["title"],
                "category":
                    exp["category"],
                "key_result":
                    result,
            }
        )

    return rows


def save_master_results(
    master,
    table_rows,
):
    """
    Save master JSON and concise CSV summary.
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_json = (
        OUTPUT_ROOT
        / "hli01_experimental_master_summary.json"
    )

    with master_json.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            master,
            file,
            indent=4,
        )

    master_csv = (
        OUTPUT_ROOT
        / "hli01_experimental_master_summary.csv"
    )

    with master_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "title",
                "category",
                "key_result",
            ],
        )

        writer.writeheader()
        writer.writerows(
            table_rows
        )

    return {
        "json": master_json,
        "csv": master_csv,
    }


def save_markdown_summary(
    master,
    table_rows,
):
    """
    Save a concise human-readable research summary.
    """

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        OUTPUT_ROOT
        / "hli01_experimental_master_summary.md"
    )

    lines = [
        "# HLI-01 v1.0.0 Experimental Validation Summary",
        "",
        "## Overview",
        "",
        (
            "This document consolidates the validated "
            "experimental evidence from EXP-001 through "
            "EXP-010."
        ),
        "",
        "## Master Results",
        "",
        "| Experiment | Study | Category | Key Result |",
        "|---|---|---|---|",
    ]

    for row in table_rows:
        lines.append(
            f"| {row['experiment']} "
            f"| {row['title']} "
            f"| {row['category']} "
            f"| {row['key_result']} |"
        )

    lines.extend(
        [
            "",
            "## Principal Findings",
            "",
            (
                "1. The baseline model achieved perfect "
                "test performance in EXP-001, while "
                "multi-seed validation in EXP-002 showed "
                "mean accuracy of approximately 99.33%."
            ),
            "",
            (
                "2. Attention produced no test-performance "
                "difference in the single-seed EXP-003 "
                "comparison, while EXP-004 showed a modest "
                "average advantage across five paired seeds."
            ),
            "",
            (
                "3. Gaussian perturbation experiments "
                "showed no measurable degradation through "
                "40% of empirical input standard deviation. "
                "The first observed mean degradation occurred "
                "at 60%."
            ),
            "",
            (
                "4. EXP-007 identified the strict sequence "
                "reset policy as a major system-level "
                "vulnerability under intermittent landmark "
                "detection."
            ),
            "",
            (
                "5. EXP-008 showed that temporal recovery "
                "policies substantially improve sequence "
                "acquisition robustness. Two-miss tolerance "
                "retained 99.9% success at 70% detection "
                "probability."
            ),
            "",
            (
                "6. EXP-009 demonstrated that the classifier "
                "itself remains comparatively robust to "
                "temporal information loss, retaining about "
                "94.33% accuracy with half of the sequence "
                "frames masked."
            ),
            "",
            (
                "7. EXP-010 demonstrated sub-millisecond "
                "mean classifier inference latency on both "
                "CPU and CUDA under the tested environment."
            ),
            "",
            "## Interpretation",
            "",
            (
                "The combined evidence suggests that HLI-01 "
                "has a highly accurate and computationally "
                "efficient classifier, with strong tolerance "
                "to moderate coordinate and temporal "
                "perturbations. The more pronounced weakness "
                "lies in the real-time temporal acquisition "
                "policy rather than the classifier itself."
            ),
            "",
            (
                "The two-miss temporal recovery strategy is "
                "therefore a promising design candidate for "
                "future real-time inference refinement, "
                "subject to validation on longer raw "
                "real-time recordings."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return path
