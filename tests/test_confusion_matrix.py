"""
=========================================================
Testing Confusion Matrix Plotter
=========================================================
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from src.visualization.confusion_matrix_plot import (
    ConfusionMatrixPlotter
)


def test_confusion_matrix():

    print("=" * 60)
    print("Testing Confusion Matrix Plotter")
    print("=" * 60)

    class_names = [

        "hello",

        "no",

        "peace",

        "yes"

    ]

    cm = np.array(

        [

            [96, 2, 1, 1],

            [3, 94, 2, 1],

            [0, 2, 97, 1],

            [1, 1, 2, 96]

        ]

    )

    plotter = ConfusionMatrixPlotter()

    image = plotter.plot(

        cm,

        class_names,

        normalize=False,

        filename="confusion_matrix.png"

    )

    print()

    print("Image Saved:")

    print(image)

    assert os.path.exists(image), \
        "Confusion matrix image not generated."

    print()

    print("✓ Confusion matrix created.")

    print()

    print("=" * 60)

    print("TEST PASSED")

    print("=" * 60)


if __name__ == "__main__":
    test_<test_confusion_matrix>()
