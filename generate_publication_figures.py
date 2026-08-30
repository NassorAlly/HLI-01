"""
Generate publication-quality figures for HLI-01 EXP-001 to EXP-012.

Outputs:
    outputs/publication_figures/

Figures:
    Figure 1 - Multi-seed classification reproducibility
    Figure 2 - Paired attention ablation
    Figure 3 - Gaussian perturbation robustness
    Figure 4 - Landmark acquisition and recovery-policy robustness
    Figure 5 - Temporal frame masking robustness
    Figure 6 - Computational efficiency
    Figure 7 - Natural end-to-end validation
    Figure 8 - Controlled interruption recovery

All figures are generated directly from authoritative experiment outputs.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path("outputs")
EXP_ROOT = ROOT / "experiments"
OUT_DIR = ROOT / "publication_figures"

STATS_JSON = (
    ROOT
    / "statistical_synthesis"
    / "hli01_statistical_summary.json"
)

EXP004_DIR = (
    EXP_ROOT
    / "EXP_20260827_213730_exp004_attention_ablation_summary"
)

EXP005_DIR = (
    EXP_ROOT
    / "EXP_20260827_220940_exp005_input_robustness"
)

EXP006_DIR = (
    EXP_ROOT
    / "EXP_20260827_223417_exp006_robustness_threshold"
)

EXP007_DIR = (
    EXP_ROOT
    / "EXP_20260827_224733_exp007_detection_interruption"
)

# IMPORTANT:
# This is the authoritative final EXP-008 run.
EXP008_DIR = (
    EXP_ROOT
    / "EXP_20260827_230254_exp008_buffer_recovery_ablation"
)

EXP009_DIR = (
    EXP_ROOT
    / "EXP_20260827_232045_exp009_temporal_masking_robustness"
)

EXP010_DIR = (
    EXP_ROOT
    / "EXP_20260827_233944_exp010_inference_efficiency"
)

EXP011_DIR = (
    EXP_ROOT
    / "EXP_20260829_215013_exp011_realtime_recovery"
)

EXP012_DIR = (
    EXP_ROOT
    / "EXP_20260830_044455_exp012_controlled_interruption"
)


def configure_matplotlib():
    """Apply a restrained, journal-friendly plotting configuration."""

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_inputs():
    """Ensure all authoritative input files exist."""

    required = [
        STATS_JSON,
        EXP004_DIR / "exp004_runs.csv",
        EXP004_DIR / "exp004_paired_deltas.csv",
        EXP005_DIR / "exp005_robustness_runs.csv",
        EXP006_DIR / "exp006_threshold_runs.csv",
        EXP007_DIR / "exp007_summary.csv",
        EXP008_DIR / "exp008_summary.csv",
        EXP009_DIR / "exp009_masking_runs.csv",
        EXP010_DIR / "exp010_efficiency_summary.csv",
        EXP011_DIR / "exp011_paired_deltas.csv",
        EXP011_DIR / "exp011_policy_results.csv",
        EXP012_DIR / "exp012_paired_deltas.csv",
        EXP012_DIR / "exp012_policy_results.csv",
    ]

    missing = [path for path in required if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing required publication-figure inputs:\n"
            + "\n".join(str(path) for path in missing)
        )

    # Explicit guard against accidentally switching to obsolete EXP-008.
    if "230254" not in str(EXP008_DIR):
        raise RuntimeError(
            "EXP-008 input must use the authoritative 230254 run."
        )


def load_statistics():
    with STATS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        exp["experiment"]: exp
        for exp in data["experiments"]
    }


def save_figure(fig, stem):
    """Save each figure in raster and vector formats."""

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUT_DIR / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_DIR / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_DIR / f"{stem}.svg",
        bbox_inches="tight",
    )

    plt.close(fig)


def bounded_ci(mean, low, high):
    """Bound confidence intervals to valid probability limits."""

    low = max(0.0, low)
    high = min(1.0, high)

    return [
        [mean - low],
        [high - mean],
    ]


def find_exp002_metrics():
    """Locate authoritative five multi-seed EXP-002 metric files."""

    patterns = [
        ("42", "EXP_20260820_003057_multiseed_seed_42"),
        ("7", "EXP_20260820_003130_multiseed_seed_7"),
        ("21", "EXP_20260820_003207_multiseed_seed_21"),
        ("84", "EXP_20260820_003236_multiseed_seed_84"),
        ("123", "EXP_20260820_003323_multiseed_seed_123"),
    ]

    rows = []

    for seed, directory in patterns:
        path = EXP_ROOT / directory / "metrics.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing EXP-002 metrics file: {path}"
            )

        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)

        def metric_value(*possible_names):
            for name in possible_names:
                if name in metrics:
                    return metrics[name]

            for container_name in (
                "test_metrics",
                "metrics",
                "final_metrics",
            ):
                container = metrics.get(container_name)
                if isinstance(container, dict):
                    for name in possible_names:
                        if name in container:
                            return container[name]

            raise KeyError(
                f"Could not find any of {possible_names} in {path}"
            )

        rows.append(
            {
                "seed": int(seed),
                "accuracy": float(
                    metric_value(
                        "test_accuracy",
                        "accuracy",
                    )
                ),
                "f1_score": float(
                    metric_value(
                        "test_f1_score",
                        "f1_score",
                        "f1",
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def figure1_multiseed(stats):
    """
    EXP-002:
    Show actual seed-level accuracy and F1 observations together with
    validated summary means.
    """

    df = find_exp002_metrics()

    x = np.arange(len(df))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    ax.bar(
        x - width / 2,
        df["accuracy"],
        width,
        label="Accuracy",
        edgecolor="black",
        linewidth=0.6,
    )

    ax.bar(
        x + width / 2,
        df["f1_score"],
        width,
        label="Weighted F1",
        edgecolor="black",
        linewidth=0.6,
    )

    exp = stats["EXP-002"]

    ax.axhline(
        exp["accuracy"]["mean"],
        linestyle="--",
        linewidth=1.1,
        label=(
            "Mean accuracy "
            f"{exp['accuracy']['mean']:.3f}"
        ),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Seed {seed}" for seed in df["seed"]]
    )

    ax.set_ylim(0.94, 1.01)
    ax.set_ylabel("Performance")
    ax.set_xlabel("Random seed")
    ax.set_title(
        "EXP-002: Multi-seed classification reproducibility"
    )

    ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(
        fig,
        "figure1_multiseed_performance",
    )


def figure2_attention_ablation(stats):
    """
    EXP-004:
    Paired seed-level accuracy for attention ON vs OFF.
    Small horizontal offsets expose overlapping observations.
    """

    df = pd.read_csv(
        EXP004_DIR / "exp004_paired_deltas.csv"
    ).sort_values("seed")

    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    offsets = np.linspace(-0.045, 0.045, len(df))

    for offset, (_, row) in zip(offsets, df.iterrows()):
        x1 = 0 + offset
        x2 = 1 + offset

        y1 = row["accuracy_attention_on"]
        y2 = row["accuracy_attention_off"]

        ax.plot(
            [x1, x2],
            [y1, y2],
            marker="o",
            linewidth=1.0,
            alpha=0.8,
        )

        label_offsets = {
            42: 0.0024,
            7: 0.0012,
            21: 0.0024,
            84: 0.0012,
            123: 0.0002,
        }

        seed = int(row["seed"])

        ax.text(
            x1 - 0.025,
            y1 + label_offsets[seed],
            str(seed),
            fontsize=7.5,
            ha="right",
            va="center",
        )

    exp = stats["EXP-004"]
    delta = exp["accuracy_delta"]

    text = (
        f"Mean paired Δ = {delta['mean']:+.4f}\n"
        f"95% CI [{delta['ci95_low']:+.4f}, "
        f"{delta['ci95_high']:+.4f}]\n"
        f"Wilcoxon p = "
        f"{exp['accuracy_wilcoxon']['p_value']:.3f}"
    )

    ax.text(
        0.03,
        0.05,
        text,
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
    )

    ax.set_xlim(-0.25, 1.25)
    ax.set_ylim(0.95, 1.01)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Attention ON", "Attention OFF"]
    )

    ax.set_ylabel("Test accuracy")
    ax.set_title(
        "EXP-004: Paired multi-seed attention ablation"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure2_attention_ablation",
    )


def figure3_gaussian_robustness(stats):
    """
    EXP-005 and EXP-006:
    Gaussian perturbation robustness and degradation threshold.
    """

    exp5 = stats["EXP-005"]
    exp6 = stats["EXP-006"]

    levels5 = pd.DataFrame(exp5["levels"])
    levels6 = pd.DataFrame(exp6["levels"])

    combined = pd.concat(
        [
            levels5,
            levels6[
                ~levels6["noise_fraction"].isin(
                    levels5["noise_fraction"]
                )
            ],
        ],
        ignore_index=True,
    ).sort_values("noise_fraction")

    x = combined["noise_fraction"].to_numpy()
    y = combined["accuracy_mean"].to_numpy()

    low = np.maximum(
        0.0,
        combined["accuracy_ci95_low"].to_numpy(),
    )

    high = np.minimum(
        1.0,
        combined["accuracy_ci95_high"].to_numpy(),
    )

    yerr = np.vstack(
        [
            y - low,
            high - y,
        ]
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=3,
        linewidth=1.4,
    )

    ax.axvline(
        0.6,
        linestyle="--",
        linewidth=1.0,
    )

    ax.text(
        0.62,
        0.84,
        "First observed\ndegradation",
        fontsize=8.5,
        va="bottom",
    )

    ax.set_ylim(0.75, 1.015)
    ax.set_xlim(-0.03, 2.05)

    ax.set_xlabel("Gaussian noise fraction")
    ax.set_ylabel("Mean test accuracy")

    ax.set_title(
        "EXP-005/006: Coordinate perturbation robustness"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure3_gaussian_robustness",
    )


def figure4_acquisition_recovery():
    """
    EXP-007 and EXP-008:
    Compare temporal recovery policies within authoritative EXP-008.
    EXP-007 strict reset is retained as an external reference.
    """

    exp7 = pd.read_csv(
        EXP007_DIR / "exp007_summary.csv"
    )

    exp8 = pd.read_csv(
        EXP008_DIR / "exp008_summary.csv"
    )

    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    policy_labels = {
        "full_reset": "Full reset",
        "one_miss_tolerance": "1-miss tolerance",
        "two_miss_tolerance": "2-miss tolerance",
        "skip_miss": "Skip miss",
    }

    for policy in [
        "full_reset",
        "one_miss_tolerance",
        "two_miss_tolerance",
        "skip_miss",
    ]:
        subset = (
            exp8[exp8["policy"] == policy]
            .sort_values("detection_probability")
        )

        ax.plot(
            subset["detection_probability"],
            subset["success_rate"],
            marker="o",
            linewidth=1.5,
            label=policy_labels[policy],
        )

    exp7_sorted = exp7.sort_values(
        "detection_probability"
    )

    ax.plot(
        exp7_sorted["detection_probability"],
        exp7_sorted["success_rate"],
        linestyle="--",
        linewidth=1.2,
        marker="x",
        label="Strict-reset reference (EXP-007)",
    )

    ax.set_xlim(0.69, 1.01)
    ax.set_ylim(-0.02, 1.03)

    ax.set_xlabel("Landmark detection probability")
    ax.set_ylabel("Acquisition success rate")

    ax.set_title(
        "EXP-007/008: Temporal acquisition robustness"
    )

    ax.legend(
        frameon=False,
        loc="lower right",
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure4_acquisition_recovery",
    )


def figure5_temporal_masking(stats):
    """EXP-009 temporal frame masking robustness."""

    exp = stats["EXP-009"]
    df = pd.DataFrame(exp["levels"])

    x = df["mask_fraction"].to_numpy()
    y = df["accuracy_mean"].to_numpy()

    low = np.maximum(
        0.0,
        df["accuracy_ci95_low"].to_numpy(),
    )

    high = np.minimum(
        1.0,
        df["accuracy_ci95_high"].to_numpy(),
    )

    yerr = np.vstack(
        [
            y - low,
            high - y,
        ]
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.6))

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        capsize=3,
        linewidth=1.4,
    )

    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(0.90, 1.01)

    ax.set_xlabel("Fraction of sequence frames masked")
    ax.set_ylabel("Mean test accuracy")

    ax.set_title(
        "EXP-009: Temporal frame masking robustness"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure5_temporal_masking",
    )


def figure6_efficiency():
    """EXP-010 CPU/CUDA latency comparison."""

    df = pd.read_csv(
        EXP010_DIR / "exp010_efficiency_summary.csv"
    )

    df["label"] = (
        df["device"].str.upper()
        + "\n"
        + df["path"].str.replace("_", " ", regex=False)
    )

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    bars = ax.bar(
        x,
        df["mean_ms"],
        edgecolor="black",
        linewidth=0.6,
    )

    ax.errorbar(
        x,
        df["mean_ms"],
        yerr=df["std_ms"],
        fmt="none",
        capsize=3,
        linewidth=1,
    )

    for bar, value in zip(
        bars,
        df["mean_ms"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])

    ax.set_ylabel("Mean latency (ms)")
    ax.set_title(
        "EXP-010: Inference computational efficiency"
    )

    ax.set_ylim(
        0,
        max(
            df["mean_ms"] + df["std_ms"]
        ) * 1.25,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "figure6_inference_efficiency",
    )


def figure7_natural_validation(stats):
    """
    EXP-011:
    Natural recording-level detection stability.
    """

    df = pd.read_csv(
        EXP011_DIR / "exp011_policy_results.csv"
    )

    # Detection rate is identical for both policies because
    # detection is a property of the recording/event stream.
    detection = (
        df[
            [
                "video",
                "filename",
                "ground_truth",
                "detection_rate",
            ]
        ]
        .drop_duplicates(
            subset=["video", "filename"]
        )
        .sort_values("detection_rate")
    )

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    x = np.arange(len(detection))

    ax.scatter(
        x,
        detection["detection_rate"],
        s=28,
    )

    exp = stats["EXP-011"]

    mean_rate = exp["detection_rate"]["mean"]

    ax.axhline(
        mean_rate,
        linestyle="--",
        linewidth=1.2,
        label=(
            f"Mean detection rate = "
            f"{mean_rate:.4f}"
        ),
    )

    ax.set_ylim(0.90, 1.005)
    ax.set_xlim(-1, len(detection))

    ax.set_xlabel("Controlled recording")
    ax.set_ylabel("Landmark detection rate")

    ax.set_title(
        "EXP-011: Natural end-to-end acquisition stability"
    )

    ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(
        fig,
        "figure7_natural_realtime_validation",
    )


def figure8_controlled_interruption(stats):
    """
    EXP-012:
    Recording-level recovery benefit across controlled dropout lengths.
    """

    df = pd.read_csv(
        EXP012_DIR / "exp012_paired_deltas.csv"
    )

    summary = (
        df.groupby("challenge_length")
        .agg(
            recovery_delay_reduction=(
                "recovery_delay_reduction",
                "mean",
            ),
            prediction_gain=(
                "prediction_gain",
                "mean",
            ),
            reset_reduction=(
                "reset_reduction",
                "mean",
            ),
            coverage_delta=(
                "coverage_delta",
                "mean",
            ),
        )
        .reset_index()
        .sort_values("challenge_length")
    )

    x = np.arange(len(summary))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.9))

    bars1 = ax.bar(
        x - width / 2,
        summary["recovery_delay_reduction"],
        width,
        label="Recovery advantage",
        edgecolor="black",
        linewidth=0.6,
    )

    bars2 = ax.bar(
        x + width / 2,
        summary["prediction_gain"],
        width,
        label="Prediction gain",
        edgecolor="black",
        linewidth=0.6,
    )

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.7,
                f"{bar.get_height():.0f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{int(length)} frame"
            if length == 1
            else f"{int(length)} frames"
            for length in summary[
                "challenge_length"
            ]
        ]
    )

    ax.set_ylabel("Frames")
    ax.set_xlabel("Injected detection interruption")

    ax.set_ylim(0, 34)

    ax.set_title(
        "EXP-012: Controlled interruption recovery advantage"
    )

    ax.legend(frameon=False)

    fig.tight_layout()

    save_figure(
        fig,
        "figure8_controlled_interruption_recovery",
    )


def write_manifest():
    """Write a simple figure manifest for reproducibility."""

    manifest = """# HLI-01 Publication Figures

Generated by `generate_publication_figures.py`.

## Figure 1
Multi-seed classification reproducibility (EXP-002).

## Figure 2
Paired multi-seed attention ablation (EXP-004).

## Figure 3
Gaussian coordinate perturbation robustness and failure threshold
(EXP-005 and EXP-006).

## Figure 4
Landmark acquisition robustness and temporal buffer recovery
(EXP-007 and EXP-008).

## Figure 5
Temporal frame masking robustness (EXP-009).

## Figure 6
CPU and CUDA inference efficiency (EXP-010).

## Figure 7
Natural end-to-end acquisition stability (EXP-011).

## Figure 8
Controlled end-to-end interruption recovery (EXP-012).

### Statistical interpretation

EXP-004:
The mean paired attention effect is small and its 95% confidence
interval includes zero. The figure should not be interpreted as
evidence of a statistically reliable attention advantage.

EXP-007/008:
Wilson intervals and simulation success rates describe Monte Carlo
uncertainty and simulated acquisition behaviour, not population-level
clinical uncertainty.

EXP-011:
Natural recordings produced too few post-buffer interruptions to
differentiate the policies. Zero paired differences do not establish
equivalence.

EXP-012:
The statistical unit is the recording. Short one- and two-frame
interruptions show a consistent 29-frame recovery advantage for the
two-miss policy, while the three-frame condition reaches the designed
tolerance boundary and produces no recovery advantage.
"""

    (OUT_DIR / "README.md").write_text(
        manifest,
        encoding="utf-8",
    )


def main():
    configure_matplotlib()
    validate_inputs()

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stats = load_statistics()

    print("Generating HLI-01 publication figures...")
    print()

    figure1_multiseed(stats)
    print("  Figure 1 complete")

    figure2_attention_ablation(stats)
    print("  Figure 2 complete")

    figure3_gaussian_robustness(stats)
    print("  Figure 3 complete")

    figure4_acquisition_recovery()
    print("  Figure 4 complete")

    figure5_temporal_masking(stats)
    print("  Figure 5 complete")

    figure6_efficiency()
    print("  Figure 6 complete")

    figure7_natural_validation(stats)
    print("  Figure 7 complete")

    figure8_controlled_interruption(stats)
    print("  Figure 8 complete")

    write_manifest()

    print()
    print(
        "Publication figures written to:"
    )
    print(f"  {OUT_DIR}")
    print()
    print(
        "Formats: PNG (600 dpi), PDF, SVG"
    )


if __name__ == "__main__":
    main()
