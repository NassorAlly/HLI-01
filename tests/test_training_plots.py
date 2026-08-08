"""
test_training_plots.py

Tests HLI-01 training-history visualizations.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.visualization import TrainingPlotter


def test_training_plots():

    print("=" * 60)
    print("HLI-01 Version 0.8.0")
    print("Testing Training Plotter")
    print("=" * 60)

    # --------------------------------------------------
    # Simulated training history
    # --------------------------------------------------

    history = {
        "train_loss": [
            1.20,
            0.95,
            0.74,
            0.55,
            0.40,
        ],
        "valid_loss": [
            1.30,
            1.02,
            0.82,
            0.64,
            0.49,
        ],
        "train_accuracy": [
            0.40,
            0.55,
            0.71,
            0.85,
            0.94,
        ],
        "valid_accuracy": [
            0.35,
            0.52,
            0.68,
            0.80,
            0.90,
        ],
        "learning_rate": [
            0.001000,
            0.001000,
            0.001000,
            0.000500,
            0.000500,
        ],
    }

    expected_epochs = len(
        history["train_loss"]
    )

    # --------------------------------------------------
    # Validate history structure
    # --------------------------------------------------

    assert expected_epochs > 0, (
        "Training history must contain "
        "at least one epoch."
    )

    assert len(
        history["valid_loss"]
    ) == expected_epochs

    assert len(
        history["train_accuracy"]
    ) == expected_epochs

    assert len(
        history["valid_accuracy"]
    ) == expected_epochs

    assert len(
        history["learning_rate"]
    ) == expected_epochs

    assert all(
        value >= 0
        for value in history["train_loss"]
    )

    assert all(
        value >= 0
        for value in history["valid_loss"]
    )

    assert all(
        0.0 <= value <= 1.0
        for value in history["train_accuracy"]
    )

    assert all(
        0.0 <= value <= 1.0
        for value in history["valid_accuracy"]
    )

    assert all(
        value > 0
        for value in history["learning_rate"]
    )

    # --------------------------------------------------
    # Plotter
    # --------------------------------------------------

    plotter = TrainingPlotter()

    # --------------------------------------------------
    # Individual plots
    # --------------------------------------------------

    loss_file = plotter.plot_loss(
        history["train_loss"],
        history["valid_loss"],
    )

    accuracy_file = plotter.plot_accuracy(
        history["train_accuracy"],
        history["valid_accuracy"],
    )

    learning_rate_file = (
        plotter.plot_learning_rate(
            history["learning_rate"]
        )
    )

    # --------------------------------------------------
    # Integrated plot_history()
    # --------------------------------------------------

    outputs = plotter.plot_history(
        history
    )

    # --------------------------------------------------
    # Display generated files
    # --------------------------------------------------

    print()

    print(
        "Loss Plot:"
    )
    print(
        loss_file
    )

    print()

    print(
        "Accuracy Plot:"
    )
    print(
        accuracy_file
    )

    print()

    print(
        "Learning Rate Plot:"
    )
    print(
        learning_rate_file
    )

    # --------------------------------------------------
    # Validate individual files
    # --------------------------------------------------

    generated_files = (
        ("Loss plot", loss_file),
        ("Accuracy plot", accuracy_file),
        (
            "Learning-rate plot",
            learning_rate_file,
        ),
    )

    for label, output_file in generated_files:

        assert output_file is not None, (
            f"{label} did not return "
            "an output path."
        )

        assert isinstance(
            output_file,
            (str, os.PathLike),
        ), (
            f"{label} returned an "
            "invalid path type."
        )

        assert os.path.exists(
            output_file
        ), (
            f"{label} was not generated."
        )

        assert os.path.isfile(
            output_file
        ), (
            f"{label} output is not "
            "a file."
        )

        assert os.path.getsize(
            output_file
        ) > 0, (
            f"{label} output file is empty."
        )

    # --------------------------------------------------
    # Validate plot_history() output
    # --------------------------------------------------

    assert isinstance(
        outputs,
        dict,
    )

    assert "loss" in outputs
    assert "accuracy" in outputs
    assert "learning_rate" in outputs

    assert os.path.exists(
        outputs["loss"]
    )

    assert os.path.exists(
        outputs["accuracy"]
    )

    assert os.path.exists(
        outputs["learning_rate"]
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()

    print(
        "✓ Loss plot generated."
    )

    print(
        "✓ Accuracy plot generated."
    )

    print(
        "✓ Learning-rate plot generated."
    )

    print(
        "✓ plot_history() works correctly."
    )

    print()

    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_training_plots()