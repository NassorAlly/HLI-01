"""
statistical_synthesis.py

HLI-01 v1.0.0
Statistical synthesis of EXP-001 through EXP-012.

This script performs publication-oriented statistical
synthesis of the completed experimental programme.

Principles:
- use recording/run/trial level as the statistical unit;
- avoid frame-level pseudoreplication;
- distinguish descriptive from inferential evidence;
- do not interpret non-significance as equivalence;
- do not force hypothesis tests where outcomes are
  deterministic or scientifically uninformative.
"""

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
from scipy import stats


EXPERIMENT_ROOT = Path("outputs/experiments")
RESEARCH_ROOT = Path("outputs/research_summary")
OUTPUT_ROOT = Path("outputs/statistical_synthesis")

MASTER_JSON = (
    RESEARCH_ROOT
    / "hli01_experimental_master_summary.json"
)

EXP004_PAIRED = (
    EXPERIMENT_ROOT
    / "EXP_20260827_213730_exp004_attention_ablation_summary"
    / "exp004_paired_deltas.csv"
)

EXP005_RUNS = (
    EXPERIMENT_ROOT
    / "EXP_20260827_220940_exp005_input_robustness"
    / "exp005_robustness_runs.csv"
)

EXP006_RUNS = (
    EXPERIMENT_ROOT
    / "EXP_20260827_223417_exp006_robustness_threshold"
    / "exp006_threshold_runs.csv"
)

EXP007_SUMMARY = (
    EXPERIMENT_ROOT
    / "EXP_20260827_224733_exp007_detection_interruption"
    / "exp007_summary.csv"
)

# IMPORTANT:
# This is the authoritative final EXP-008 run.
EXP008_SUMMARY = (
    EXPERIMENT_ROOT
    / "EXP_20260827_230254_exp008_buffer_recovery_ablation"
    / "exp008_summary.csv"
)

EXP009_RUNS = (
    EXPERIMENT_ROOT
    / "EXP_20260827_232045_exp009_temporal_masking_robustness"
    / "exp009_masking_runs.csv"
)

EXP010_SUMMARY = (
    EXPERIMENT_ROOT
    / "EXP_20260827_233944_exp010_inference_efficiency"
    / "exp010_efficiency_summary.csv"
)

EXP011_DETECTION = (
    EXPERIMENT_ROOT
    / "EXP_20260829_215013_exp011_realtime_recovery"
    / "exp011_detection_events_summary.csv"
)

EXP011_PAIRED = (
    EXPERIMENT_ROOT
    / "EXP_20260829_215013_exp011_realtime_recovery"
    / "exp011_paired_deltas.csv"
)

EXP012_PAIRED = (
    EXPERIMENT_ROOT
    / "EXP_20260830_044455_exp012_controlled_interruption"
    / "exp012_paired_deltas.csv"
)


def load_json(path):
    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def validate_files():
    required = [
        MASTER_JSON,
        EXP004_PAIRED,
        EXP005_RUNS,
        EXP006_RUNS,
        EXP007_SUMMARY,
        EXP008_SUMMARY,
        EXP009_RUNS,
        EXP010_SUMMARY,
        EXP011_DETECTION,
        EXP011_PAIRED,
        EXP012_PAIRED,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing statistical source files:\n"
            + "\n".join(missing)
        )


def t_confidence_interval(values, confidence=0.95):
    """
    Two-sided Student-t confidence interval for a mean.
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    if n == 0:
        return None, None

    if n == 1:
        value = float(values[0])
        return value, value

    sample_mean = float(
        np.mean(values)
    )

    sample_sem = float(
        stats.sem(values)
    )

    if sample_sem == 0.0:
        return sample_mean, sample_mean

    critical = stats.t.ppf(
        (1.0 + confidence) / 2.0,
        df=n - 1,
    )

    margin = (
        float(critical)
        * sample_sem
    )

    return (
        sample_mean - margin,
        sample_mean + margin,
    )


def wilson_interval(successes, trials, confidence=0.95):
    """
    Wilson score interval for a binomial proportion.
    """

    successes = int(successes)
    trials = int(trials)

    if trials <= 0:
        return None, None

    p = successes / trials

    z = stats.norm.ppf(
        1.0 - (1.0 - confidence) / 2.0
    )

    denominator = (
        1.0
        + (z ** 2) / trials
    )

    centre = (
        p
        + (z ** 2) / (2.0 * trials)
    ) / denominator

    half_width = (
        z
        / denominator
        * math.sqrt(
            p * (1.0 - p) / trials
            + (z ** 2)
            / (4.0 * trials ** 2)
        )
    )

    return (
        centre - half_width,
        centre + half_width,
    )


def describe(values, bounds=None):
    """
    Descriptive statistics with a Student-t confidence
    interval for the mean.

    When bounds are supplied, the raw interval is retained
    for transparency and a bounded reporting interval is
    also provided. This is useful for metrics such as
    accuracy and detection rate that are constrained to
    [0, 1].
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    ci_low, ci_high = (
        t_confidence_interval(values)
    )

    result = {
        "n":
            int(len(values)),

        "mean":
            float(np.mean(values)),

        "std":
            (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            ),

        "median":
            float(np.median(values)),

        "min":
            float(np.min(values)),

        "max":
            float(np.max(values)),

        "ci95_low":
            float(ci_low),

        "ci95_high":
            float(ci_high),
    }

    if bounds is not None:
        lower_bound, upper_bound = bounds

        result[
            "reporting_ci95_low"
        ] = float(
            max(
                lower_bound,
                ci_low,
            )
        )

        result[
            "reporting_ci95_high"
        ] = float(
            min(
                upper_bound,
                ci_high,
            )
        )

    return result


def safe_wilcoxon(differences):
    """
    Paired Wilcoxon signed-rank test.

    Zero-only differences are reported descriptively
    rather than forcing a meaningless test.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    nonzero = differences[
        differences != 0
    ]

    if len(nonzero) == 0:
        return {
            "performed": False,
            "reason":
                "All paired differences are zero.",
            "statistic": None,
            "p_value": None,
            "nonzero_pairs": 0,
        }

    result = stats.wilcoxon(
        differences,
        alternative="two-sided",
        zero_method="wilcox",
        method="auto",
    )

    return {
        "performed": True,
        "statistic":
            float(result.statistic),
        "p_value":
            float(result.pvalue),
        "nonzero_pairs":
            int(len(nonzero)),
    }


def exact_sign_test(differences):
    """
    Two-sided exact sign test based only on
    positive versus negative non-zero differences.
    """

    differences = np.asarray(
        differences,
        dtype=float,
    )

    positive = int(
        np.sum(differences > 0)
    )

    negative = int(
        np.sum(differences < 0)
    )

    n = positive + negative

    if n == 0:
        return {
            "performed": False,
            "positive": positive,
            "negative": negative,
            "nonzero_pairs": 0,
            "p_value": None,
            "reason":
                "No non-zero paired differences.",
        }

    result = stats.binomtest(
        k=min(
            positive,
            negative,
        ),
        n=n,
        p=0.5,
        alternative="two-sided",
    )

    return {
        "performed": True,
        "positive": positive,
        "negative": negative,
        "nonzero_pairs": n,
        "p_value":
            float(result.pvalue),
    }


def get_experiment(master, experiment_id):
    return next(
        experiment
        for experiment
        in master["experiments"]
        if experiment["experiment"]
        == experiment_id
    )


def synthesize_exp001(master):
    exp = get_experiment(
        master,
        "EXP-001",
    )

    return {
        "experiment": "EXP-001",
        "title": exp["title"],
        "evidence_type":
            "descriptive_single_baseline",
        "accuracy":
            exp["test_accuracy"],
        "f1_score":
            exp["test_f1_score"],
        "interpretation":
            (
                "Single baseline evaluation; "
                "reported descriptively without "
                "inferential testing."
            ),
    }


def synthesize_exp002(master):
    exp = get_experiment(
        master,
        "EXP-002",
    )

    accuracy = [
        run["accuracy"]
        for run in exp["runs"]
    ]

    f1 = [
        run["f1_score"]
        for run in exp["runs"]
    ]

    return {
        "experiment": "EXP-002",
        "title": exp["title"],
        "evidence_type":
            "multi_seed_reproducibility",
        "accuracy":
            describe(
                accuracy,
                bounds=(0.0, 1.0),
            ),
        "f1_score":
            describe(
                f1,
                bounds=(0.0, 1.0),
            ),
        "interpretation":
            (
                "Five independent seed runs quantify "
                "training reproducibility."
            ),
    }


def synthesize_exp003(master):
    exp = get_experiment(
        master,
        "EXP-003",
    )

    return {
        "experiment": "EXP-003",
        "title": exp["title"],
        "evidence_type":
            "single_seed_ablation",
        "attention_on_accuracy":
            exp["attention_on"]["accuracy"],
        "attention_off_accuracy":
            exp["attention_off"]["accuracy"],
        "accuracy_delta":
            (
                exp["attention_on"]["accuracy"]
                - exp["attention_off"]["accuracy"]
            ),
        "interpretation":
            (
                "Single-seed comparison; descriptive "
                "only. Multi-seed inference is based "
                "on EXP-004."
            ),
    }


def synthesize_exp004():
    df = pd.read_csv(
        EXP004_PAIRED
    )

    accuracy_delta = (
        df["accuracy_delta"]
        .astype(float)
        .to_numpy()
    )

    f1_delta = (
        df["f1_delta"]
        .astype(float)
        .to_numpy()
    )

    return {
        "experiment": "EXP-004",
        "title":
            "Multi-seed Paired Attention Ablation",
        "evidence_type":
            "paired_multi_seed_ablation",
        "accuracy_delta":
            describe(accuracy_delta),
        "accuracy_wilcoxon":
            safe_wilcoxon(
                accuracy_delta
            ),
        "accuracy_sign_test":
            exact_sign_test(
                accuracy_delta
            ),
        "f1_delta":
            describe(f1_delta),
        "f1_wilcoxon":
            safe_wilcoxon(
                f1_delta
            ),
        "interpretation":
            (
                "Paired seed-level analysis. "
                "The small sample size (n=5) "
                "requires cautious interpretation."
            ),
    }


def summarize_grouped_runs(
    path,
    group_column,
    metric_columns,
):
    df = pd.read_csv(path)

    rows = []

    for level, group in df.groupby(
        group_column,
        sort=True,
    ):
        row = {
            group_column:
                float(level),
            "runs":
                int(len(group)),
        }

        for metric in metric_columns:
            summary = describe(
                group[metric]
                .astype(float)
                .to_numpy()
            )

            for key, value in summary.items():
                row[
                    f"{metric}_{key}"
                ] = value

        rows.append(row)

    return rows


def synthesize_exp005():
    return {
        "experiment": "EXP-005",
        "title":
            "Gaussian Input Perturbation Robustness",
        "evidence_type":
            "repeated_perturbation_robustness",
        "levels":
            summarize_grouped_runs(
                EXP005_RUNS,
                "noise_fraction",
                [
                    "accuracy",
                    "f1_score",
                ],
            ),
        "interpretation":
            (
                "Performance is summarized across "
                "repeated perturbation seeds at each "
                "noise level."
            ),
    }


def synthesize_exp006():
    return {
        "experiment": "EXP-006",
        "title":
            "Gaussian Robustness Failure Threshold",
        "evidence_type":
            "threshold_robustness",
        "levels":
            summarize_grouped_runs(
                EXP006_RUNS,
                "noise_fraction",
                [
                    "accuracy",
                    "f1_score",
                ],
            ),
        "interpretation":
            (
                "Threshold behaviour is evaluated "
                "using mean performance and "
                "95% confidence intervals across "
                "perturbation seeds."
            ),
    }


def synthesize_exp007():
    df = pd.read_csv(
        EXP007_SUMMARY
    )

    levels = []

    for _, row in df.iterrows():
        low, high = wilson_interval(
            row["successes"],
            row["trials"],
        )

        levels.append(
            {
                "detection_probability":
                    float(
                        row[
                            "detection_probability"
                        ]
                    ),
                "trials":
                    int(row["trials"]),
                "successes":
                    int(row["successes"]),
                "success_rate":
                    float(
                        row["success_rate"]
                    ),
                "success_rate_ci95_low":
                    float(low),
                "success_rate_ci95_high":
                    float(high),
                "mean_frames_required":
                    float(
                        row[
                            "mean_frames_required"
                        ]
                    ),
                "mean_resets":
                    float(
                        row["mean_resets"]
                    ),
            }
        )

    return {
        "experiment": "EXP-007",
        "title":
            "Landmark Detection Interruption Robustness",
        "evidence_type":
            "simulation_binomial_robustness",
        "levels": levels,
        "interpretation":
            (
                "Success probabilities are accompanied "
                "by Wilson 95% confidence intervals. "
                "These intervals quantify Monte Carlo "
                "simulation uncertainty rather than "
                "population-level clinical uncertainty."
            ),
    }


def synthesize_exp008():
    df = pd.read_csv(
        EXP008_SUMMARY
    )

    levels = []

    for _, row in df.iterrows():
        low, high = wilson_interval(
            row["successes"],
            row["trials"],
        )

        levels.append(
            {
                "policy":
                    row["policy"],
                "detection_probability":
                    float(
                        row[
                            "detection_probability"
                        ]
                    ),
                "trials":
                    int(row["trials"]),
                "successes":
                    int(row["successes"]),
                "success_rate":
                    float(
                        row["success_rate"]
                    ),
                "success_rate_ci95_low":
                    float(low),
                "success_rate_ci95_high":
                    float(high),
                "mean_frames_required":
                    float(
                        row[
                            "mean_frames_required"
                        ]
                    ),
                "mean_resets":
                    float(
                        row["mean_resets"]
                    ),
                "mean_misses":
                    float(
                        row["mean_misses"]
                    ),
            }
        )

    comparisons = []

    probabilities = sorted(
        df[
            "detection_probability"
        ].unique(),
        reverse=True,
    )

    for probability in probabilities:
        subset = df[
            df[
                "detection_probability"
            ] == probability
        ]

        control = subset[
            subset["policy"]
            == "full_reset"
        ]

        if control.empty:
            continue

        control_rate = float(
            control.iloc[0][
                "success_rate"
            ]
        )

        for _, row in subset.iterrows():
            if row["policy"] == "full_reset":
                continue

            comparisons.append(
                {
                    "detection_probability":
                        float(probability),
                    "policy":
                        row["policy"],
                    "success_rate":
                        float(
                            row[
                                "success_rate"
                            ]
                        ),
                    "full_reset_success_rate":
                        control_rate,
                    "absolute_success_gain":
                        float(
                            row[
                                "success_rate"
                            ]
                            - control_rate
                        ),
                }
            )

    return {
        "experiment": "EXP-008",
        "title":
            "Temporal Buffer Recovery Strategy Ablation",
        "evidence_type":
            "simulation_policy_comparison",
        "levels": levels,
        "comparisons_to_full_reset":
            comparisons,
        "interpretation":
            (
                "Policy effects are reported as "
                "absolute success-rate improvements "
                "relative to full reset. No population "
                "inference is claimed from the "
                "simulation trials."
            ),
    }


def synthesize_exp009():
    return {
        "experiment": "EXP-009",
        "title":
            "Temporal Frame Masking Robustness",
        "evidence_type":
            "repeated_temporal_perturbation",
        "levels":
            summarize_grouped_runs(
                EXP009_RUNS,
                "mask_fraction",
                [
                    "accuracy",
                    "f1_score",
                ],
            ),
        "interpretation":
            (
                "Temporal masking robustness is "
                "summarized across repeated mask "
                "seeds using mean, SD and 95% CI."
            ),
    }


def synthesize_exp010():
    df = pd.read_csv(
        EXP010_SUMMARY
    )

    benchmarks = []

    for _, row in df.iterrows():
        benchmarks.append(
            {
                "device":
                    row["device"],
                "path":
                    row["path"],
                "mean_ms":
                    float(row["mean_ms"]),
                "median_ms":
                    float(
                        row["median_ms"]
                    ),
                "std_ms":
                    float(row["std_ms"]),
                "p95_ms":
                    float(row["p95_ms"]),
                "p99_ms":
                    float(row["p99_ms"]),
                "sequences_per_second":
                    float(
                        row[
                            "sequences_per_second"
                        ]
                    ),
            }
        )

    return {
        "experiment": "EXP-010",
        "title":
            "Inference Performance and Computational Efficiency",
        "evidence_type":
            "latency_benchmark",
        "benchmarks":
            benchmarks,
        "interpretation":
            (
                "Latency is treated as benchmark "
                "performance and summarized "
                "descriptively."
            ),
    }


def synthesize_exp011():
    detection = pd.read_csv(
        EXP011_DETECTION
    )

    paired = pd.read_csv(
        EXP011_PAIRED
    )

    detection_rates = (
        detection["detection_rate"]
        .astype(float)
        .to_numpy()
    )

    prediction_gain = (
        paired["prediction_gain"]
        .astype(float)
        .to_numpy()
    )

    reset_reduction = (
        paired["reset_reduction"]
        .astype(float)
        .to_numpy()
    )

    return {
        "experiment": "EXP-011",
        "title":
            "End-to-End Real-Time Temporal Recovery Validation",
        "evidence_type":
            "paired_natural_condition_validation",
        "recordings":
            int(len(paired)),
        "detection_rate":
            describe(
                detection_rates,
                bounds=(0.0, 1.0),
            ),
        "prediction_gain":
            describe(
                prediction_gain
            ),
        "reset_reduction":
            describe(
                reset_reduction
            ),
        "prediction_gain_test":
            safe_wilcoxon(
                prediction_gain
            ),
        "interpretation":
            (
                "Natural interruptions did not "
                "meaningfully challenge the established "
                "temporal buffer. Zero paired policy "
                "differences therefore do not establish "
                "statistical equivalence."
            ),
    }


def synthesize_exp012():
    df = pd.read_csv(
        EXP012_PAIRED
    )

    conditions = {}

    for condition, group in df.groupby(
        "condition",
        sort=False,
    ):
        recovery = (
            group[
                "recovery_delay_reduction"
            ]
            .astype(float)
            .to_numpy()
        )

        reset = (
            group["reset_reduction"]
            .astype(float)
            .to_numpy()
        )

        prediction = (
            group["prediction_gain"]
            .astype(float)
            .to_numpy()
        )

        coverage = (
            group["coverage_delta"]
            .astype(float)
            .to_numpy()
        )

        full_correct = (
            group[
                "full_reset_video_correct"
            ]
            .astype(str)
            .str.lower()
            .eq("true")
            .to_numpy()
        )

        two_correct = (
            group[
                "two_miss_video_correct"
            ]
            .astype(str)
            .str.lower()
            .eq("true")
            .to_numpy()
        )

        discordant = int(
            np.sum(
                full_correct
                != two_correct
            )
        )

        conditions[
            condition
        ] = {
            "recordings":
                int(len(group)),

            "challenge_length":
                int(
                    group[
                        "challenge_length"
                    ].iloc[0]
                ),

            "recovery_delay_reduction":
                describe(recovery),

            "recovery_wilcoxon":
                safe_wilcoxon(
                    recovery
                ),

            "recovery_sign_test":
                exact_sign_test(
                    recovery
                ),

            "reset_reduction":
                describe(reset),

            "prediction_gain":
                describe(prediction),

            "coverage_delta":
                describe(coverage),

            "full_reset_video_accuracy":
                float(
                    np.mean(
                        full_correct
                    )
                ),

            "two_miss_video_accuracy":
                float(
                    np.mean(
                        two_correct
                    )
                ),

            "video_accuracy_discordant_pairs":
                discordant,

            "mcnemar_performed":
                False,

            "mcnemar_reason":
                (
                    "Not performed because there "
                    "were no discordant video-level "
                    "classification outcomes."
                    if discordant == 0
                    else
                    "Discordant pairs exist; "
                    "McNemar analysis could be "
                    "considered."
                ),
        }

    return {
        "experiment": "EXP-012",
        "title":
            "Controlled End-to-End Detection Interruption Challenge",
        "evidence_type":
            "paired_controlled_interruption",
        "statistical_unit":
            "recording",
        "conditions":
            conditions,
        "interpretation":
            (
                "Primary inference is based on "
                "recording-level paired recovery "
                "effects. Frame-level observations "
                "are not treated as independent "
                "replicates. Classification accuracy "
                "remained identical between policies, "
                "so superiority testing of accuracy "
                "is not informative."
            ),
    }


def build_synthesis():
    master = load_json(
        MASTER_JSON
    )

    experiments = [
        synthesize_exp001(master),
        synthesize_exp002(master),
        synthesize_exp003(master),
        synthesize_exp004(),
        synthesize_exp005(),
        synthesize_exp006(),
        synthesize_exp007(),
        synthesize_exp008(),
        synthesize_exp009(),
        synthesize_exp010(),
        synthesize_exp011(),
        synthesize_exp012(),
    ]

    return {
        "project": "HLI-01",
        "version": "v1.0.0",
        "analysis_scope":
            "EXP-001 through EXP-012",
        "number_of_experiments":
            len(experiments),
        "statistical_principles": [
            (
                "Statistical unit follows the "
                "experimental design."
            ),
            (
                "Frame-level predictions are not "
                "treated as independent replicates "
                "for end-to-end video experiments."
            ),
            (
                "Confidence intervals and effect "
                "magnitudes are prioritized over "
                "isolated p-values."
            ),
            (
                "Non-significant findings are not "
                "interpreted as evidence of "
                "equivalence."
            ),
            (
                "Simulation confidence intervals "
                "describe Monte Carlo uncertainty."
            ),
        ],
        "experiments":
            experiments,
    }


def build_summary_rows(synthesis):
    rows = []

    for exp in synthesis[
        "experiments"
    ]:
        experiment = exp[
            "experiment"
        ]

        if experiment == "EXP-001":
            result = (
                f"Accuracy={exp['accuracy']:.4f}; "
                f"F1={exp['f1_score']:.4f}"
            )

        elif experiment == "EXP-002":
            result = (
                f"Accuracy="
                f"{exp['accuracy']['mean']:.4f} "
                f"(95% CI "
                f"{exp['accuracy']['reporting_ci95_low']:.4f}"
                f"–"
                f"{exp['accuracy']['reporting_ci95_high']:.4f})"
            )

        elif experiment == "EXP-003":
            result = (
                f"Single-seed accuracy delta="
                f"{exp['accuracy_delta']:+.4f}"
            )

        elif experiment == "EXP-004":
            result = (
                f"Mean paired accuracy delta="
                f"{exp['accuracy_delta']['mean']:+.4f}; "
                f"n={exp['accuracy_delta']['n']}"
            )

        elif experiment in {
            "EXP-005",
            "EXP-006",
            "EXP-009",
        }:
            strongest = exp[
                "levels"
            ][-1]

            result = (
                f"Strongest tested level: "
                f"accuracy="
                f"{strongest['accuracy_mean']:.4f} "
                f"(95% CI "
                f"{strongest['accuracy_ci95_low']:.4f}"
                f"–"
                f"{strongest['accuracy_ci95_high']:.4f})"
            )

        elif experiment == "EXP-007":
            level_90 = next(
                level
                for level in exp["levels"]
                if math.isclose(
                    level[
                        "detection_probability"
                    ],
                    0.9,
                )
            )

            result = (
                f"90% detection success="
                f"{level_90['success_rate']:.3f} "
                f"(95% Wilson CI "
                f"{level_90['success_rate_ci95_low']:.3f}"
                f"–"
                f"{level_90['success_rate_ci95_high']:.3f})"
            )

        elif experiment == "EXP-008":
            target = next(
                row
                for row in exp[
                    "comparisons_to_full_reset"
                ]
                if (
                    row["policy"]
                    == "two_miss_tolerance"
                    and math.isclose(
                        row[
                            "detection_probability"
                        ],
                        0.7,
                    )
                )
            )

            result = (
                f"Two-miss success at 70%="
                f"{target['success_rate']:.3f}; "
                f"absolute gain vs full reset="
                f"{target['absolute_success_gain']:+.3f}"
            )

        elif experiment == "EXP-010":
            raw_cpu = next(
                row
                for row in exp[
                    "benchmarks"
                ]
                if (
                    row["device"] == "cpu"
                    and row["path"]
                    == "raw_model"
                )
            )

            result = (
                f"CPU raw-model latency="
                f"{raw_cpu['mean_ms']:.4f} ms "
                f"(p95={raw_cpu['p95_ms']:.4f} ms)"
            )

        elif experiment == "EXP-011":
            result = (
                f"Mean detection rate="
                f"{exp['detection_rate']['mean']:.4f}; "
                f"paired prediction gain="
                f"{exp['prediction_gain']['mean']:+.4f}"
            )

        elif experiment == "EXP-012":
            one = exp[
                "conditions"
            ][
                "dropout_1_frame"
            ]

            two = exp[
                "conditions"
            ][
                "dropout_2_frames"
            ]

            three = exp[
                "conditions"
            ][
                "dropout_3_frames"
            ]

            result = (
                f"Mean recovery advantage: "
                f"{one['recovery_delay_reduction']['mean']:+.0f}, "
                f"{two['recovery_delay_reduction']['mean']:+.0f}, "
                f"{three['recovery_delay_reduction']['mean']:+.0f} "
                f"frames for 1/2/3-frame dropouts"
            )

        else:
            result = ""

        rows.append(
            {
                "experiment":
                    experiment,
                "title":
                    exp["title"],
                "evidence_type":
                    exp[
                        "evidence_type"
                    ],
                "statistical_summary":
                    result,
            }
        )

    return rows


def save_outputs(
    synthesis,
    rows,
):
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        OUTPUT_ROOT
        / "hli01_statistical_summary.json"
    )

    csv_path = (
        OUTPUT_ROOT
        / "hli01_statistical_summary.csv"
    )

    md_path = (
        OUTPUT_ROOT
        / "hli01_statistical_report.md"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            synthesis,
            file,
            indent=4,
        )

    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "experiment",
                "title",
                "evidence_type",
                "statistical_summary",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# HLI-01 v1.0.0 Statistical Synthesis",
        "",
        "## Scope",
        "",
        (
            "This report synthesizes statistical "
            "evidence from EXP-001 through EXP-012."
        ),
        "",
        "## Statistical Principles",
        "",
        (
            "- Experimental runs, simulation trials, "
            "or recordings are used as statistical "
            "units according to the study design."
        ),
        (
            "- Frame-level predictions from the same "
            "recording are not treated as independent "
            "replicates."
        ),
        (
            "- Effect magnitudes and confidence "
            "intervals are emphasized."
        ),
        (
            "- Non-significant results are not "
            "interpreted as equivalence."
        ),
        "",
        "## Experiment-Level Summary",
        "",
        (
            "| Experiment | Evidence Type | "
            "Statistical Summary |"
        ),
        "|---|---|---|",
    ]

    for row in rows:
        lines.append(
            f"| {row['experiment']} "
            f"| {row['evidence_type']} "
            f"| {row['statistical_summary']} |"
        )

    exp004 = next(
        exp
        for exp in synthesis[
            "experiments"
        ]
        if exp["experiment"]
        == "EXP-004"
    )

    exp011 = next(
        exp
        for exp in synthesis[
            "experiments"
        ]
        if exp["experiment"]
        == "EXP-011"
    )

    exp012 = next(
        exp
        for exp in synthesis[
            "experiments"
        ]
        if exp["experiment"]
        == "EXP-012"
    )

    lines.extend(
        [
            "",
            "## Inferential Interpretation",
            "",
            (
                "### EXP-004 — Attention Ablation"
            ),
            "",
            (
                f"The mean paired accuracy difference "
                f"(attention ON minus OFF) was "
                f"{exp004['accuracy_delta']['mean']:+.4f} "
                f"across "
                f"{exp004['accuracy_delta']['n']} paired "
                f"seeds. Because only five paired seeds "
                f"were available, inferential results "
                f"should be interpreted cautiously."
            ),
            "",
            (
                "### EXP-011 — Natural Real-Time Validation"
            ),
            "",
            (
                f"The mean natural landmark-detection "
                f"rate was "
                f"{exp011['detection_rate']['mean']:.4f}. "
                f"The mean paired prediction gain was "
                f"{exp011['prediction_gain']['mean']:+.4f}. "
                f"However, the absence of a policy "
                f"difference does not demonstrate "
                f"equivalence because natural "
                f"post-buffer interruptions were not "
                f"sufficiently represented."
            ),
            "",
            (
                "### EXP-012 — Controlled Interruption Challenge"
            ),
            "",
        ]
    )

    for condition, result in exp012[
        "conditions"
    ].items():
        recovery = result[
            "recovery_delay_reduction"
        ]

        sign = result[
            "recovery_sign_test"
        ]

        if sign["performed"]:
            sign_text = (
                f"exact sign-test p="
                f"{sign['p_value']:.3e}"
            )
        else:
            sign_text = (
                "no sign test was performed because "
                "all paired differences were zero"
            )

        lines.append(
            (
                f"- **{condition}**: mean recovery-delay "
                f"reduction="
                f"{recovery['mean']:+.2f} frames "
                f"(n={recovery['n']}); "
                f"{sign_text}. "
                f"Video accuracy remained "
                f"{result['full_reset_video_accuracy']:.4f} "
                f"for full reset and "
                f"{result['two_miss_video_accuracy']:.4f} "
                f"for two-miss tolerance."
            )
        )

    lines.extend(
        [
            "",
            "## Overall Statistical Conclusion",
            "",
            (
                "The complete experimental programme "
                "supports three distinct conclusions. "
                "First, HLI-01 classification performance "
                "is highly reproducible across seeds and "
                "remains robust under moderate coordinate "
                "and temporal perturbation. Second, the "
                "strict full-reset acquisition policy is "
                "substantially more vulnerable to "
                "intermittent landmark loss than the "
                "classifier itself. Third, controlled "
                "end-to-end validation demonstrates that "
                "two-miss tolerance materially reduces "
                "recovery delay and increases prediction "
                "availability for one- and two-frame "
                "interruptions while preserving "
                "classification accuracy. At the "
                "three-frame interruption boundary, its "
                "behaviour converges to the conservative "
                "full-reset policy as designed."
            ),
            "",
            (
                "The EXP-012 significance results should "
                "be interpreted together with the effect "
                "magnitudes and deterministic challenge "
                "design rather than as isolated evidence "
                "from p-values."
            ),
            "",
        ]
    )

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
    }


def main():
    print("=" * 64)
    print(
        "HLI-01 Statistical Synthesis: EXP-001 through EXP-012"
    )
    print("=" * 64)

    validate_files()

    synthesis = build_synthesis()

    rows = build_summary_rows(
        synthesis
    )

    outputs = save_outputs(
        synthesis,
        rows,
    )

    print()
    print(
        "Experiments synthesized :",
        synthesis[
            "number_of_experiments"
        ],
    )

    print(
        "Statistical JSON        :",
        outputs["json"],
    )

    print(
        "Statistical CSV         :",
        outputs["csv"],
    )

    print(
        "Statistical report      :",
        outputs["markdown"],
    )

    print()
    print(
        "Statistical synthesis completed successfully."
    )


if __name__ == "__main__":
    main()
