"""
=========================================================
HLI-01 Version 0.7.0
Testing Training Plotter
=========================================================
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.visualization import TrainingPlotter


def test_training_plots():

    print("=" * 60)
    print("HLI-01 Version 0.7.0")
    print("Testing Training Plotter")
    print("=" * 60)

    train_loss = [1.20, 0.95, 0.74, 0.55, 0.40]
    val_loss = [1.30, 1.02, 0.82, 0.64, 0.49]

    train_accuracy = [40, 55, 71, 85, 94]
    val_accuracy = [35, 52, 68, 80, 90]

    expected_epochs = len(train_loss)

    assert expected_epochs > 0, (
        "Training history must contain at least one epoch."
    )

    assert len(val_loss) == expected_epochs, (
        "Training and validation loss must have equal lengths."
    )

    assert len(train_accuracy) == expected_epochs, (
        "Training loss and training accuracy must have equal lengths."
    )

    assert len(val_accuracy) == expected_epochs, (
        "Training loss and validation accuracy must have equal lengths."
    )

    assert all(value >= 0 for value in train_loss), (
        "Training loss must not contain negative values."
    )

    assert all(value >= 0 for value in val_loss), (
        "Validation loss must not contain negative values."
    )

    assert all(0 <= value <= 100 for value in train_accuracy), (
        "Training accuracy must be between 0 and 100."
    )

    assert all(0 <= value <= 100 for value in val_accuracy), (
        "Validation accuracy must be between 0 and 100."
    )

    plotter = TrainingPlotter()

    loss_file = plotter.plot_loss(
        train_loss,
        val_loss,
    )

    accuracy_file = plotter.plot_accuracy(
        train_accuracy,
        val_accuracy,
    )

    print()
    print("Loss Plot:")
    print(loss_file)

    print()
    print("Accuracy Plot:")
    print(accuracy_file)

    for label, output_file in (
        ("Loss plot", loss_file),
        ("Accuracy plot", accuracy_file),
    ):
        assert output_file is not None, (
            f"{label} did not return an output path."
        )

        assert isinstance(output_file, (str, os.PathLike)), (
            f"{label} returned an invalid path type."
        )

        assert os.path.exists(output_file), (
            f"{label} was not generated."
        )

        assert os.path.isfile(output_file), (
            f"{label} output is not a file."
        )

        assert os.path.getsize(output_file) > 0, (
            f"{label} output file is empty."
        )

    print()
    print("✓ Loss plot generated")
    print("✓ Accuracy plot generated")
    print()
    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_training_plots()
