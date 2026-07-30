"""
=========================================================
HLI-01 Version 0.6.0
Unit Test - Evaluator
=========================================================
"""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.evaluator import Evaluator


def test_evaluator():

    print("=" * 60)
    print("HLI-01 Version 0.6.0")
    print("Unit Test - Evaluator")
    print("=" * 60)

    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 2, 0, 0, 2]

    assert len(y_true) == len(y_pred), (
        "True and predicted label lists must have equal lengths."
    )

    assert len(y_true) > 0, (
        "Evaluation data must not be empty."
    )

    results = Evaluator.evaluate(y_true, y_pred)

    required_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "confusion_matrix",
    }

    assert isinstance(results, dict), (
        "Evaluator must return a dictionary."
    )

    assert required_keys.issubset(results.keys()), (
        "Evaluator result is missing required metrics."
    )

    for metric_name in (
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    ):
        metric_value = results[metric_name]

        assert isinstance(metric_value, (int, float)), (
            f"{metric_name} must be numeric."
        )

        assert 0.0 <= metric_value <= 1.0, (
            f"{metric_name} must be between 0 and 1."
        )

    expected_accuracy = 5 / 6

    assert np.isclose(
        results["accuracy"],
        expected_accuracy,
        atol=1e-6,
    ), (
        "Evaluator returned an incorrect accuracy."
    )

    confusion_matrix = results["confusion_matrix"]

    assert isinstance(confusion_matrix, np.ndarray), (
        "Confusion matrix must be a NumPy array."
    )

    assert confusion_matrix.shape == (3, 3), (
        "Confusion matrix must have shape (3, 3)."
    )

    assert confusion_matrix.sum() == len(y_true), (
        "Confusion matrix total must match the number of samples."
    )

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-Score : {results['f1_score']:.4f}")

    print("\nConfusion Matrix")
    print(confusion_matrix)

    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_evaluator()