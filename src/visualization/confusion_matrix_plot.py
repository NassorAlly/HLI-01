"""
=========================================================
HLI-01 Version 0.6.0
Confusion Matrix Visualization
=========================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from .visualization_utils import create_directory


class ConfusionMatrixPlotter:
    """
    Plot and save confusion matrices.
    """

    def __init__(self, save_dir="outputs/confusion_matrix"):

        self.save_dir = save_dir

        create_directory(self.save_dir)

    def plot(
        self,
        confusion_matrix,
        class_names,
        normalize=False,
        filename="confusion_matrix.png"
    ):
        """
        Plot a confusion matrix.

        Parameters
        ----------
        confusion_matrix : ndarray
            NxN confusion matrix

        class_names : list
            Class labels

        normalize : bool
            Display percentages if True

        filename : str
            Output filename
        """

        cm = np.array(confusion_matrix, dtype=float)

        if normalize:

            row_sum = cm.sum(axis=1, keepdims=True)

            row_sum[row_sum == 0] = 1

            cm = cm / row_sum

        plt.figure(figsize=(7, 6))

        plt.imshow(cm, interpolation="nearest", cmap="Blues")

        plt.title("Confusion Matrix")

        plt.colorbar()

        tick_marks = np.arange(len(class_names))

        plt.xticks(tick_marks, class_names, rotation=45)

        plt.yticks(tick_marks, class_names)

        threshold = cm.max() / 2

        for i in range(cm.shape[0]):

            for j in range(cm.shape[1]):

                if normalize:

                    text = f"{cm[i, j]:.2f}"

                else:

                    text = str(int(cm[i, j]))

                plt.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black"
                )

        plt.ylabel("True Label")

        plt.xlabel("Predicted Label")

        plt.tight_layout()

        filepath = os.path.join(
            self.save_dir,
            filename
        )

        plt.savefig(filepath, dpi=300)

        plt.close()

        return filepath
