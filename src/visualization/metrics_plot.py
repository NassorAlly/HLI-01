"""
=========================================================
HLI-01 Version 0.6.0
Classification Metrics Visualization
=========================================================
"""

import os
import matplotlib.pyplot as plt

from .visualization_utils import create_directory


class MetricsPlotter:
    """
    Creates a single figure containing Precision,
    Recall, F1-Score and Support.
    """

    def __init__(self, save_dir="outputs/metrics"):

        self.save_dir = save_dir

        create_directory(self.save_dir)

    def plot(self, metrics, class_names,
             filename="classification_metrics.png"):

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(12, 8)
        )

        plots = [

            ("Precision",
             metrics["precision"],
             axes[0, 0],
             True),

            ("Recall",
             metrics["recall"],
             axes[0, 1],
             True),

            ("F1 Score",
             metrics["f1_score"],
             axes[1, 0],
             True),

            ("Support",
             metrics["support"],
             axes[1, 1],
             False)

        ]

        for title, values, ax, normalized in plots:

            ax.bar(class_names, values)

            ax.set_title(title)

            ax.set_xlabel("Class")

            if normalized:

                ax.set_ylim(0, 1.05)

                ax.set_ylabel("Score")

            else:

                ax.set_ylabel("Samples")

            ax.grid(axis="y")

        plt.suptitle(
            "Classification Metrics",
            fontsize=16,
            fontweight="bold"
        )

        plt.tight_layout()

        filepath = os.path.join(
            self.save_dir,
            filename
        )

        plt.savefig(
            filepath,
            dpi=300
        )

        plt.close()

        return filepath
