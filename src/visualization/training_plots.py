"""
training_plots.py

Visualization utilities for HLI-01 training history.
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

    def __init__(
        self,
        save_dir="outputs/figures",
    ):

        self.save_dir = save_dir

        create_directory(
            self.save_dir
        )

    def plot_loss(
        self,
        train_loss,
        val_loss,
    ):
        """
        Plot training and validation loss.
        """

        epochs = range(
            1,
            len(train_loss) + 1,
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            epochs,
            train_loss,
            marker="o",
            linewidth=2,
            label="Training Loss",
        )

        plt.plot(
            epochs,
            val_loss,
            marker="s",
            linewidth=2,
            label="Validation Loss",
        )

        plt.title(
            "Training and Validation Loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Loss"
        )

        plt.grid(
            True
        )

        plt.legend()

        plt.tight_layout()

        filename = os.path.join(
            self.save_dir,
            "loss_curve.png",
        )

        plt.savefig(
            filename
        )

        plt.close()

        return filename

    def plot_accuracy(
        self,
        train_accuracy,
        val_accuracy,
    ):
        """
        Plot training and validation accuracy.
        """

        epochs = range(
            1,
            len(train_accuracy) + 1,
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            epochs,
            train_accuracy,
            marker="o",
            linewidth=2,
            label="Training Accuracy",
        )

        plt.plot(
            epochs,
            val_accuracy,
            marker="s",
            linewidth=2,
            label="Validation Accuracy",
        )

        plt.title(
            "Training and Validation Accuracy"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Accuracy"
        )

        plt.grid(
            True
        )

        plt.legend()

        plt.tight_layout()

        filename = os.path.join(
            self.save_dir,
            "accuracy_curve.png",
        )

        plt.savefig(
            filename
        )

        plt.close()

        return filename

    def plot_learning_rate(
        self,
        learning_rate,
    ):
        """
        Plot learning-rate history.
        """

        epochs = range(
            1,
            len(learning_rate) + 1,
        )

        plt.figure(
            figsize=(8, 5)
        )

        plt.plot(
            epochs,
            learning_rate,
            marker="o",
            linewidth=2,
        )

        plt.title(
            "Learning Rate Schedule"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "Learning Rate"
        )

        plt.grid(
            True
        )

        plt.tight_layout()

        filename = os.path.join(
            self.save_dir,
            "learning_rate_curve.png",
        )

        plt.savefig(
            filename
        )

        plt.close()

        return filename

    def plot_history(
        self,
        history,
    ):
        """
        Generate all training-history plots.

        Parameters
        ----------
        history : dict
            Trainer history dictionary.

        Returns
        -------
        dict
            Paths to generated figures.
        """

        required_keys = [
            "train_loss",
            "train_accuracy",
            "valid_loss",
            "valid_accuracy",
        ]

        for key in required_keys:

            if key not in history:

                raise KeyError(
                    f"Missing history key: {key}"
                )

        outputs = {}

        outputs["loss"] = (
            self.plot_loss(
                history["train_loss"],
                history["valid_loss"],
            )
        )

        outputs["accuracy"] = (
            self.plot_accuracy(
                history["train_accuracy"],
                history["valid_accuracy"],
            )
        )

        if (
            "learning_rate" in history
            and len(
                history["learning_rate"]
            ) > 0
        ):

            outputs["learning_rate"] = (
                self.plot_learning_rate(
                    history["learning_rate"]
                )
            )

        return outputs