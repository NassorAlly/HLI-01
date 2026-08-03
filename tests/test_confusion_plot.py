"""
=========================================================
HLI-01 Version 0.7.0
Testing Confusion Matrix Plotter
=========================================================
"""

import os
import sys

import numpy as np


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from src.visualization.confusion_matrix_plot import ConfusionMatrixPlotter


def test_confusion_plot():

    print("=" * 60)
    print("HLI-01 Version 0.7.0")
    print("Testing Confusion Matrix Plotter")
    print("=" * 60)

    class_names = [
        "hello",
        "no",
        "peace",
        "yes",
    ]

    confusion_matrix = np.array(
        [
            [96, 2, 1, 1],
            [3, 94, 2, 1],
            [0, 2, 97, 1],
            [1, 1, 2, 96],
        ]
    )

    assert confusion_matrix.shape == (4, 4)
    assert len(class_names) == confusion_matrix.shape[0]

    plotter = ConfusionMatrixPlotter()

    image = plotter.plot(
        confusion_matrix,
        class_names,
        normalize=False,
        filename="confusion_matrix.png",
    )

    print()
    print("Confusion Matrix Saved:")
    print(image)
    print()

    assert image is not None, "The plotter did not return an output path."
    assert isinstance(image, (str, os.PathLike)), (
        "The returned image path has an invalid type."
    )
    assert os.path.exists(image), (
        "Confusion matrix image was not generated."
    )
    assert os.path.getsize(image) > 0, (
        "The generated confusion matrix image is empty."
    )

    print("✓ Confusion matrix generated successfully")
    print()
    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_confusion_plot()
