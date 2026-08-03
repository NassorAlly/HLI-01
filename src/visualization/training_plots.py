"""
=========================================================
HLI-01 Version 0.7.0
Training Visualization Module
=========================================================
"""

import os

import matplotlib

# Use a non-interactive backend suitable for automated testing
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .visualization_utils import create_directory


class TrainingPlotter:
    """
    Plot training history.
    """

    def __init__(self, save_dir="outputs/figures"):

        self.save_dir = save_dir

        create_directory(self.save_dir)

    def plot_loss(self, train_loss, val_loss):

        plt.figure(figsize=(8, 5))

        plt.plot(
            train_loss,
            marker="o",
            linewidth=2,
            label="Training Loss"
        )

        plt.plot(
            val_loss,
            marker="s",
            linewidth=2,
            label="Validation Loss"
        )

        plt.title("Training Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        filename = os.path.join(
            self.save_dir,
            "loss_curve.png"
        )

        plt.savefig(filename)

        plt.close()

        return filename

    def plot_accuracy(self, train_accuracy, val_accuracy):

        plt.figure(figsize=(8, 5))

        plt.plot(
            train_accuracy,
            marker="o",
            linewidth=2,
            label="Training Accuracy"
        )

        plt.plot(
            val_accuracy,
            marker="s",
            linewidth=2,
            label="Validation Accuracy"
        )

        plt.title("Training Accuracy")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy (%)")

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        filename = os.path.join(
            self.save_dir,
            "accuracy_curve.png"
        )

        plt.savefig(filename)

        plt.close()

        return filename
