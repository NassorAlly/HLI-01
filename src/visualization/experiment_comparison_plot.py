from pathlib import Path
from typing import Dict, List, Any

import matplotlib.pyplot as plt


class ExperimentComparisonPlotter:
    """
    Create publication-ready comparison plots for HLI-01 experiments.
    """

    def __init__(self, save_dir="outputs/comparisons"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def plot(
        self,
        experiments: List[Dict[str, Any]],
        metrics: List[str],
        filename: str = "experiment_comparison.png",
    ) -> str:
        if not experiments:
            raise ValueError(
                "Experiments cannot be empty."
            )

        if not metrics:
            raise ValueError(
                "Metrics cannot be empty."
            )

        names = [
            experiment["name"]
            for experiment in experiments
        ]

        x_positions = list(range(len(names)))

        width = 0.8 / len(metrics)

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for index, metric in enumerate(metrics):
            values = []

            for experiment in experiments:
                if metric not in experiment:
                    raise KeyError(
                        f"Metric '{metric}' missing from experiment."
                    )

                values.append(
                    experiment[metric]
                )

            offset = (
                index - (len(metrics) - 1) / 2
            ) * width

            positions = [
                x + offset
                for x in x_positions
            ]

            ax.bar(
                positions,
                values,
                width=width,
                label=metric,
            )

        ax.set_title(
            "HLI-01 Experiment Comparison"
        )

        ax.set_xlabel("Experiment")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            names,
            rotation=20,
            ha="right",
        )

        ax.legend()
        ax.grid(
            axis="y",
            alpha=0.3,
        )

        fig.tight_layout()

        output_path = (
            self.save_dir / filename
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)

        return str(output_path)
