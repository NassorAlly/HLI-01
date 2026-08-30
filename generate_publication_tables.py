"""
generate_publication_tables.py

HLI-01 v1.0.0
Publication-ready tables for EXP-001 through EXP-012.

Inputs:
    outputs/statistical_synthesis/hli01_statistical_summary.json

Outputs:
    outputs/publication_tables/
        table1_experimental_programme.csv
        table1_experimental_programme.md
        table2_classification_attention.csv
        table2_classification_attention.md
        table3_robustness.csv
        table3_robustness.md
        table4_efficiency.csv
        table4_efficiency.md
        table5_realtime_recovery.csv
        table5_realtime_recovery.md
        publication_tables.md
"""

import csv
import json
from pathlib import Path


INPUT_JSON = Path(
    "outputs/statistical_synthesis/"
    "hli01_statistical_summary.json"
)

OUTPUT_ROOT = Path(
    "outputs/publication_tables"
)


def load_data():
    with INPUT_JSON.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def get_exp(data, experiment_id):
    return next(
        exp
        for exp in data["experiments"]
        if exp["experiment"] == experiment_id
    )


def fmt4(value):
    return f"{float(value):.4f}"


def fmt3(value):
    return f"{float(value):.3f}"


def fmt_pct(value):
    return f"{100.0 * float(value):.2f}%"


def save_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def markdown_escape(value):
    return str(value).replace("|", "\\|")


def save_markdown(path, title, rows):
    if not rows:
        return

    headers = list(rows[0].keys())

    lines = [
        f"# {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(
            ["---"] * len(headers)
        ) + " |",
    ]

    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                markdown_escape(
                    row[h]
                )
                for h in headers
            )
            + " |"
        )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_table1(data):
    """
    Table 1:
    Overall experimental validation programme.
    """

    purposes = {
        "EXP-001":
            "Establish reference classification performance",

        "EXP-002":
            "Evaluate reproducibility across random seeds",

        "EXP-003":
            "Initial attention-component ablation",

        "EXP-004":
            "Evaluate attention contribution across paired seeds",

        "EXP-005":
            "Assess robustness to moderate coordinate perturbation",

        "EXP-006":
            "Identify coordinate-noise degradation threshold",

        "EXP-007":
            "Assess effect of intermittent landmark detection loss",

        "EXP-008":
            "Compare temporal buffer recovery policies",

        "EXP-009":
            "Assess robustness to temporal frame masking",

        "EXP-010":
            "Measure inference latency and computational efficiency",

        "EXP-011":
            "Validate temporal recovery under natural recording conditions",

        "EXP-012":
            "Validate recovery policy under controlled end-to-end interruptions",
    }

    evidence = {
        "EXP-001":
            "Single baseline evaluation",

        "EXP-002":
            "Five-seed reproducibility study",

        "EXP-003":
            "Single-seed ablation",

        "EXP-004":
            "Five-seed paired ablation",

        "EXP-005":
            "Repeated Gaussian perturbation",

        "EXP-006":
            "Extended perturbation threshold analysis",

        "EXP-007":
            "Monte Carlo acquisition simulation",

        "EXP-008":
            "Monte Carlo policy ablation",

        "EXP-009":
            "Repeated temporal masking",

        "EXP-010":
            "CPU/GPU latency benchmark",

        "EXP-011":
            "40 controlled raw recordings",

        "EXP-012":
            "40 recordings × 3 controlled interruption conditions",
    }

    key_results = {}

    exp1 = get_exp(data, "EXP-001")
    key_results["EXP-001"] = (
        f"Accuracy {fmt_pct(exp1['accuracy'])}; "
        f"F1 {fmt_pct(exp1['f1_score'])}"
    )

    exp2 = get_exp(data, "EXP-002")
    key_results["EXP-002"] = (
        f"Mean accuracy "
        f"{fmt_pct(exp2['accuracy']['mean'])} "
        f"(SD {fmt_pct(exp2['accuracy']['std'])})"
    )

    exp3 = get_exp(data, "EXP-003")
    key_results["EXP-003"] = (
        f"Accuracy difference "
        f"{100 * exp3['accuracy_delta']:+.2f} percentage points"
    )

    exp4 = get_exp(data, "EXP-004")
    key_results["EXP-004"] = (
        f"Mean paired accuracy difference "
        f"{100 * exp4['accuracy_delta']['mean']:+.2f} percentage points; "
        f"Wilcoxon p={exp4['accuracy_wilcoxon']['p_value']:.3f}"
    )

    key_results["EXP-005"] = (
        "Accuracy remained 100% through "
        "noise fraction 0.40"
    )

    key_results["EXP-006"] = (
        "First degradation at noise fraction 0.60; "
        "accuracy 82% at fraction 2.00"
    )

    exp7 = get_exp(data, "EXP-007")
    level90 = next(
        row
        for row in exp7["levels"]
        if abs(
            row["detection_probability"] - 0.90
        ) < 1e-9
    )
    key_results["EXP-007"] = (
        f"Success rate {fmt_pct(level90['success_rate'])} "
        "at 90% landmark-detection probability"
    )

    exp8 = get_exp(data, "EXP-008")
    target = next(
        row
        for row in exp8[
            "comparisons_to_full_reset"
        ]
        if (
            row["policy"]
            == "two_miss_tolerance"
            and abs(
                row["detection_probability"] - 0.70
            ) < 1e-9
        )
    )
    key_results["EXP-008"] = (
        f"Two-miss success {fmt_pct(target['success_rate'])} "
        "at 70% detection probability"
    )

    key_results["EXP-009"] = (
        "Accuracy 94.33% with 50% of frames masked"
    )

    exp10 = get_exp(data, "EXP-010")
    cpu_raw = next(
        row
        for row in exp10["benchmarks"]
        if (
            row["device"] == "cpu"
            and row["path"] == "raw_model"
        )
    )
    cuda_raw = next(
        row
        for row in exp10["benchmarks"]
        if (
            row["device"] == "cuda"
            and row["path"] == "raw_model"
        )
    )
    key_results["EXP-010"] = (
        f"Mean raw-model latency "
        f"{cpu_raw['mean_ms']:.3f} ms CPU; "
        f"{cuda_raw['mean_ms']:.3f} ms CUDA"
    )

    exp11 = get_exp(data, "EXP-011")
    key_results["EXP-011"] = (
        f"Mean detection rate "
        f"{fmt_pct(exp11['detection_rate']['mean'])}; "
        "100% video accuracy"
    )

    exp12 = get_exp(data, "EXP-012")
    one = exp12["conditions"]["dropout_1_frame"]
    two = exp12["conditions"]["dropout_2_frames"]
    three = exp12["conditions"]["dropout_3_frames"]

    key_results["EXP-012"] = (
        f"Recovery advantage "
        f"{one['recovery_delay_reduction']['mean']:.0f}/"
        f"{two['recovery_delay_reduction']['mean']:.0f}/"
        f"{three['recovery_delay_reduction']['mean']:.0f} frames "
        "for 1/2/3-frame interruptions"
    )

    rows = []

    for experiment_id in [
        f"EXP-{i:03d}"
        for i in range(1, 13)
    ]:
        exp = get_exp(
            data,
            experiment_id,
        )

        rows.append(
            {
                "Experiment":
                    experiment_id,

                "Study objective":
                    purposes[
                        experiment_id
                    ],

                "Design":
                    evidence[
                        experiment_id
                    ],

                "Key result":
                    key_results[
                        experiment_id
                    ],
            }
        )

    return rows


def build_table2(data):
    """
    Table 2:
    Core classification and attention results.
    """

    exp1 = get_exp(data, "EXP-001")
    exp2 = get_exp(data, "EXP-002")
    exp3 = get_exp(data, "EXP-003")
    exp4 = get_exp(data, "EXP-004")

    rows = [
        {
            "Experiment":
                "EXP-001",

            "Analysis":
                "Baseline",

            "n":
                "1 run",

            "Accuracy":
                fmt4(exp1["accuracy"]),

            "F1 score":
                fmt4(exp1["f1_score"]),

            "95% CI / paired effect":
                "Not applicable",

            "Interpretation":
                "Reference performance",
        },

        {
            "Experiment":
                "EXP-002",

            "Analysis":
                "Multi-seed reproducibility",

            "n":
                str(
                    exp2[
                        "accuracy"
                    ]["n"]
                ),

            "Accuracy":
                (
                    f"{fmt4(exp2['accuracy']['mean'])} "
                    f"± "
                    f"{fmt4(exp2['accuracy']['std'])}"
                ),

            "F1 score":
                (
                    f"{fmt4(exp2['f1_score']['mean'])} "
                    f"± "
                    f"{fmt4(exp2['f1_score']['std'])}"
                ),

            "95% CI / paired effect":
                (
                    f"Accuracy "
                    f"{fmt4(exp2['accuracy']['reporting_ci95_low'])}"
                    f"–"
                    f"{fmt4(exp2['accuracy']['reporting_ci95_high'])}"
                ),

            "Interpretation":
                "High reproducibility across seeds",
        },

        {
            "Experiment":
                "EXP-003",

            "Analysis":
                "Single-seed attention ablation",

            "n":
                "1 paired seed",

            "Accuracy":
                (
                    f"ON "
                    f"{fmt4(exp3['attention_on_accuracy'])}; "
                    f"OFF "
                    f"{fmt4(exp3['attention_off_accuracy'])}"
                ),

            "F1 score":
                "—",

            "95% CI / paired effect":
                (
                    f"ΔAcc "
                    f"{exp3['accuracy_delta']:+.4f}"
                ),

            "Interpretation":
                "Descriptive only",
        },

        {
            "Experiment":
                "EXP-004",

            "Analysis":
                "Multi-seed paired attention ablation",

            "n":
                str(
                    exp4[
                        "accuracy_delta"
                    ]["n"]
                ),

            "Accuracy":
                "Paired ON vs OFF",

            "F1 score":
                (
                    f"Mean Δ "
                    f"{exp4['f1_delta']['mean']:+.4f}"
                ),

            "95% CI / paired effect":
                (
                    f"Mean ΔAcc "
                    f"{exp4['accuracy_delta']['mean']:+.4f}; "
                    f"95% CI "
                    f"{exp4['accuracy_delta']['ci95_low']:+.4f}"
                    f" to "
                    f"{exp4['accuracy_delta']['ci95_high']:+.4f}; "
                    f"Wilcoxon p="
                    f"{exp4['accuracy_wilcoxon']['p_value']:.3f}"
                ),

            "Interpretation":
                (
                    "Small positive mean difference; "
                    "95% CI includes zero"
                ),
        },
    ]

    return rows


def build_table3(data):
    """
    Table 3:
    Robustness results.
    """

    exp5 = get_exp(data, "EXP-005")
    exp6 = get_exp(data, "EXP-006")
    exp7 = get_exp(data, "EXP-007")
    exp8 = get_exp(data, "EXP-008")
    exp9 = get_exp(data, "EXP-009")

    exp5_04 = next(
        row
        for row in exp5["levels"]
        if abs(
            row["noise_fraction"] - 0.40
        ) < 1e-9
    )

    exp6_06 = next(
        row
        for row in exp6["levels"]
        if abs(
            row["noise_fraction"] - 0.60
        ) < 1e-9
    )

    exp6_20 = next(
        row
        for row in exp6["levels"]
        if abs(
            row["noise_fraction"] - 2.00
        ) < 1e-9
    )

    exp7_09 = next(
        row
        for row in exp7["levels"]
        if abs(
            row["detection_probability"] - 0.90
        ) < 1e-9
    )

    exp8_07 = next(
        row
        for row in exp8[
            "comparisons_to_full_reset"
        ]
        if (
            row["policy"]
            == "two_miss_tolerance"
            and abs(
                row["detection_probability"] - 0.70
            ) < 1e-9
        )
    )

    exp9_05 = next(
        row
        for row in exp9["levels"]
        if abs(
            row["mask_fraction"] - 0.50
        ) < 1e-9
    )

    rows = [
        {
            "Experiment":
                "EXP-005",

            "Perturbation":
                "Gaussian coordinate noise",

            "Evaluation point":
                "Noise fraction 0.40",

            "Primary result":
                (
                    f"Accuracy "
                    f"{fmt4(exp5_04['accuracy_mean'])}"
                ),

            "Uncertainty / effect":
                (
                    f"95% CI "
                    f"{fmt4(exp5_04['accuracy_ci95_low'])}"
                    f"–"
                    f"{fmt4(exp5_04['accuracy_ci95_high'])}"
                ),

            "Conclusion":
                "No observed degradation through moderate noise",
        },

        {
            "Experiment":
                "EXP-006",

            "Perturbation":
                "Extended Gaussian coordinate noise",

            "Evaluation point":
                "Noise fractions 0.60 and 2.00",

            "Primary result":
                (
                    f"First degradation at 0.60; "
                    f"accuracy at 2.00 = "
                    f"{fmt4(exp6_20['accuracy_mean'])}"
                ),

            "Uncertainty / effect":
                (
                    f"Accuracy at 0.60 = "
                    f"{fmt4(exp6_06['accuracy_mean'])}"
                ),

            "Conclusion":
                "Identified robustness failure threshold",
        },

        {
            "Experiment":
                "EXP-007",

            "Perturbation":
                "Intermittent landmark detection",

            "Evaluation point":
                "Detection probability 0.90",

            "Primary result":
                (
                    f"Acquisition success "
                    f"{fmt4(exp7_09['success_rate'])}"
                ),

            "Uncertainty / effect":
                (
                    f"95% Wilson CI "
                    f"{fmt4(exp7_09['success_rate_ci95_low'])}"
                    f"–"
                    f"{fmt4(exp7_09['success_rate_ci95_high'])}"
                ),

            "Conclusion":
                "Strict reset policy is sensitive to detection loss",
        },

        {
            "Experiment":
                "EXP-008",

            "Perturbation":
                "Temporal buffer recovery policy",

            "Evaluation point":
                "Detection probability 0.70",

            "Primary result":
                (
                    f"Two-miss success "
                    f"{fmt4(exp8_07['success_rate'])}"
                ),

            "Uncertainty / effect":
                (
                    f"Absolute gain vs full reset "
                    f"{exp8_07['absolute_success_gain']:+.4f}"
                ),

            "Conclusion":
                "Two-miss tolerance markedly improves simulated acquisition robustness",
        },

        {
            "Experiment":
                "EXP-009",

            "Perturbation":
                "Temporal frame masking",

            "Evaluation point":
                "50% frames masked",

            "Primary result":
                (
                    f"Accuracy "
                    f"{fmt4(exp9_05['accuracy_mean'])}"
                ),

            "Uncertainty / effect":
                (
                    f"95% CI "
                    f"{fmt4(exp9_05['accuracy_ci95_low'])}"
                    f"–"
                    f"{fmt4(exp9_05['accuracy_ci95_high'])}"
                ),

            "Conclusion":
                "Classifier retains substantial temporal robustness",
        },
    ]

    return rows


def build_table4(data):
    """
    Table 4:
    Computational efficiency.
    """

    exp10 = get_exp(data, "EXP-010")

    rows = []

    for result in exp10["benchmarks"]:
        rows.append(
            {
                "Device":
                    result["device"].upper(),

                "Inference path":
                    result["path"],

                "Mean latency (ms)":
                    f"{result['mean_ms']:.4f}",

                "Median latency (ms)":
                    f"{result['median_ms']:.4f}",

                "P95 latency (ms)":
                    f"{result['p95_ms']:.4f}",

                "P99 latency (ms)":
                    f"{result['p99_ms']:.4f}",

                "Sequences/s":
                    f"{result['sequences_per_second']:.1f}",
            }
        )

    return rows


def build_table5(data):
    """
    Table 5:
    End-to-end natural and controlled temporal recovery.
    """

    exp11 = get_exp(data, "EXP-011")
    exp12 = get_exp(data, "EXP-012")

    rows = [
        {
            "Experiment / condition":
                "EXP-011 natural recordings",

            "n":
                str(exp11["recordings"]),

            "Mean detection rate":
                fmt4(
                    exp11[
                        "detection_rate"
                    ]["mean"]
                ),

            "Reset reduction":
                fmt4(
                    exp11[
                        "reset_reduction"
                    ]["mean"]
                ),

            "Prediction gain":
                fmt4(
                    exp11[
                        "prediction_gain"
                    ]["mean"]
                ),

            "Recovery advantage (frames)":
                "Not estimable",

            "Video accuracy":
                "1.0000",

            "Interpretation":
                (
                    "Natural post-buffer interruptions "
                    "were too rare to differentiate policies"
                ),
        }
    ]

    mapping = [
        (
            "dropout_1_frame",
            "EXP-012: 1-frame interruption",
        ),
        (
            "dropout_2_frames",
            "EXP-012: 2-frame interruption",
        ),
        (
            "dropout_3_frames",
            "EXP-012: 3-frame interruption",
        ),
    ]

    for condition_key, label in mapping:
        condition = exp12[
            "conditions"
        ][condition_key]

        recovery = condition[
            "recovery_delay_reduction"
        ]

        reset = condition[
            "reset_reduction"
        ]

        prediction = condition[
            "prediction_gain"
        ]


        rows.append(
            {
                "Experiment / condition":
                    label,

                "n":
                    str(
                        condition[
                            "recordings"
                        ]
                    ),

                "Mean detection rate":
                    "—",

                "Reset reduction":
                    f"{reset['mean']:.2f}",

                "Prediction gain":
                    f"{prediction['mean']:.2f}",

                "Recovery advantage (frames)":
                    (
                        f"{recovery['mean']:.2f}; "
                        f"{condition['recovery_sign_test']['positive']}/"
                        f"{condition['recordings']} recordings improved"
                        if condition["recovery_sign_test"]["performed"]
                        else
                        f"{recovery['mean']:.2f}; "
                        "all paired differences zero"
                    ),

                "Video accuracy":
                    (
                        f"{condition['full_reset_video_accuracy']:.4f} "
                        "vs "
                        f"{condition['two_miss_video_accuracy']:.4f}"
                    ),

                "Interpretation":
                    (
                        "Two-miss tolerance improves "
                        "short-interruption recovery"
                        if condition[
                            "challenge_length"
                        ] <= 2
                        else
                        (
                            "Designed boundary reached; "
                            "two-miss policy matches full reset"
                        )
                    ),
            }
        )

    return rows


def main():
    print("=" * 68)
    print(
        "HLI-01 Publication Tables: EXP-001 through EXP-012"
    )
    print("=" * 68)

    if not INPUT_JSON.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_JSON}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    tables = [
        (
            "table1_experimental_programme",
            "Table 1. HLI-01 Experimental Validation Programme",
            build_table1(data),
        ),
        (
            "table2_classification_attention",
            "Table 2. Classification Performance and Attention Analysis",
            build_table2(data),
        ),
        (
            "table3_robustness",
            "Table 3. Robustness Evaluation",
            build_table3(data),
        ),
        (
            "table4_efficiency",
            "Table 4. Computational Efficiency",
            build_table4(data),
        ),
        (
            "table5_realtime_recovery",
            "Table 5. End-to-End Temporal Recovery Validation",
            build_table5(data),
        ),
    ]

    combined = [
        "# HLI-01 v1.0.0 Publication Tables",
        "",
    ]

    for stem, title, rows in tables:
        csv_path = (
            OUTPUT_ROOT
            / f"{stem}.csv"
        )

        md_path = (
            OUTPUT_ROOT
            / f"{stem}.md"
        )

        save_csv(
            csv_path,
            rows,
        )

        save_markdown(
            md_path,
            title,
            rows,
        )

        combined.append(
            md_path.read_text(
                encoding="utf-8"
            )
        )

        combined.append("")

        print(
            f"Generated: {csv_path}"
        )
        print(
            f"Generated: {md_path}"
        )

    combined_path = (
        OUTPUT_ROOT
        / "publication_tables.md"
    )

    combined_path.write_text(
        "\n".join(combined),
        encoding="utf-8",
    )

    print()
    print(
        "Combined publication tables:",
        combined_path,
    )

    print()
    print(
        "Publication table generation completed successfully."
    )


if __name__ == "__main__":
    main()
