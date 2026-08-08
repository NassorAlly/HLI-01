"""
evaluate.py

Final test-set evaluation entry point for HLI-01 v0.8.0.

Loads the best trained checkpoint, evaluates it on the
held-out test dataset, displays the results, and saves
the official evaluation results to disk.
"""

from pathlib import Path

import numpy as np
import torch

from src.config.settings import (
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_CLASSES,
    LEARNING_RATE,
    CHECKPOINT_DIR,
    BEST_MODEL_FILENAME,
)

from src.models.lstm_model import LSTMModel
from src.training.data_loader import DataLoaderManager
from src.training.checkpoint_manager import CheckpointManager
from src.evaluation.evaluator import Evaluator


# =====================================================
# OUTPUT CONFIGURATION
# =====================================================

RESULTS_DIR = Path("outputs/evaluation")

RESULTS_FILE = (
    RESULTS_DIR / "evaluation_results.txt"
)

MATRIX_FILE = (
    RESULTS_DIR / "confusion_matrix.npy"
)


def save_results(
    results,
    epoch,
    validation_loss,
    num_samples,
):
    """
    Save official evaluation results.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    confusion_matrix = results[
        "confusion_matrix"
    ]

    # Save confusion matrix as NumPy array
    np.save(
        MATRIX_FILE,
        confusion_matrix,
    )

    # Save human-readable evaluation report
    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "HLI-01 v0.8.0\n"
        )

        file.write(
            "Final Test-Set Evaluation\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Best Epoch          : "
            f"{epoch + 1}\n"
        )

        file.write(
            f"Best Validation Loss: "
            f"{validation_loss:.4f}\n"
        )

        file.write(
            f"Test Samples        : "
            f"{num_samples}\n\n"
        )

        file.write(
            "Classification Metrics\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"Accuracy     : "
            f"{results['accuracy']:.4f}\n"
        )

        file.write(
            f"Precision    : "
            f"{results['precision']:.4f}\n"
        )

        file.write(
            f"Recall       : "
            f"{results['recall']:.4f}\n"
        )

        file.write(
            f"F1-score     : "
            f"{results['f1_score']:.4f}\n\n"
        )

        file.write(
            "Percentage Metrics\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            f"Accuracy     : "
            f"{results['accuracy'] * 100:.2f}%\n"
        )

        file.write(
            f"Precision    : "
            f"{results['precision'] * 100:.2f}%\n"
        )

        file.write(
            f"Recall       : "
            f"{results['recall'] * 100:.2f}%\n"
        )

        file.write(
            f"F1-score     : "
            f"{results['f1_score'] * 100:.2f}%\n\n"
        )

        file.write(
            "Confusion Matrix\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        file.write(
            np.array2string(
                confusion_matrix
            )
        )

        file.write("\n")


def main():
    """
    Evaluate the best HLI-01 checkpoint
    on the held-out test dataset.
    """

    print("=" * 60)
    print(
        "HLI-01 v0.8.0 - Final Test Evaluation"
    )
    print("=" * 60)

    # =================================================
    # DEVICE
    # =================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "Device:",
        device,
    )

    # =================================================
    # TEST DATA
    # =================================================

    print()
    print(
        "Loading test dataset..."
    )

    data_manager = (
        DataLoaderManager()
    )

    (
        _,
        _,
        test_loader,
    ) = data_manager.create()

    print(
        "Testing batches:",
        len(test_loader),
    )

    # =================================================
    # MODEL
    # =================================================

    print()
    print(
        "Creating model..."
    )

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
    )

    model = model.to(
        device
    )

    # =================================================
    # OPTIMIZER
    # =================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # =================================================
    # CHECKPOINT
    # =================================================

    print()
    print(
        "Loading best checkpoint..."
    )

    checkpoint_manager = (
        CheckpointManager(
            checkpoint_dir=CHECKPOINT_DIR
        )
    )

    epoch, validation_loss = (
        checkpoint_manager.load(
            model=model,
            optimizer=optimizer,
            filename=BEST_MODEL_FILENAME,
            device=device,
        )
    )

    print(
        "Best epoch          :",
        epoch + 1,
    )

    print(
        "Validation loss     :",
        f"{validation_loss:.4f}",
    )

    # =================================================
    # TEST INFERENCE
    # =================================================

    print()
    print(
        "Running test-set inference..."
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(
                device
            )

            labels = labels.to(
                device
            )

            outputs = model(
                inputs
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            y_true.extend(
                labels.cpu().numpy()
            )

            y_pred.extend(
                predictions.cpu().numpy()
            )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    # =================================================
    # EVALUATION
    # =================================================

    results = Evaluator.evaluate(
        y_true,
        y_pred,
    )

    # =================================================
    # DISPLAY RESULTS
    # =================================================

    print()
    print("=" * 60)
    print("Final Test Results")
    print("=" * 60)

    print()

    print(
        "Test samples :",
        len(y_true),
    )

    print(
        "Accuracy     :",
        f"{results['accuracy']:.4f}",
    )

    print(
        "Precision    :",
        f"{results['precision']:.4f}",
    )

    print(
        "Recall       :",
        f"{results['recall']:.4f}",
    )

    print(
        "F1-score     :",
        f"{results['f1_score']:.4f}",
    )

    print()

    print(
        "Accuracy (%) :",
        f"{results['accuracy'] * 100:.2f}%",
    )

    print(
        "Precision (%):",
        f"{results['precision'] * 100:.2f}%",
    )

    print(
        "Recall (%)   :",
        f"{results['recall'] * 100:.2f}%",
    )

    print(
        "F1-score (%) :",
        f"{results['f1_score'] * 100:.2f}%",
    )

    # =================================================
    # CONFUSION MATRIX
    # =================================================

    print()
    print("=" * 60)
    print("Confusion Matrix")
    print("=" * 60)

    print()

    print(
        results["confusion_matrix"]
    )

    # =================================================
    # SAVE RESULTS
    # =================================================

    save_results(
        results=results,
        epoch=epoch,
        validation_loss=validation_loss,
        num_samples=len(y_true),
    )

    print()
    print("=" * 60)
    print("Evaluation Outputs")
    print("=" * 60)

    print()

    print(
        "Evaluation report :",
        RESULTS_FILE,
    )

    print(
        "Confusion matrix  :",
        MATRIX_FILE,
    )

    # =================================================
    # FINAL STATUS
    # =================================================

    print()
    print("=" * 60)

    print(
        "HLI-01 v0.8.0 evaluation "
        "completed successfully."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()