"""
=========================================================
HLI-01 Version 0.7.0
Unit Test - Evaluation Metrics
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

from src.evaluation.metrics import Metrics


def test_metrics():
    """Test the Metrics class."""

    print("=" * 60)
    print("HLI-01 Version 0.7.0")
    print("Unit Test - Evaluation Metrics")
    print("=" * 60)

    y_true = [0, 1, 2, 1, 0, 2]
    y_pred = [0, 1, 2, 0, 0, 2]

    assert len(y_true) == len(y_pred), (
        "True and predicted label lists must have equal lengths."
    )

    assert len(y_true) > 0, (
        "Evaluation data must not be empty."
    )

    results = Metrics.evaluate(y_true, y_pred)

    assert isinstance(results, dict), (
        "Metrics.evaluate() must return a dictionary."
    )

    required_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
    }

    assert required_keys.issubset(results.keys()), (
        "Metrics result is missing one or more required values."
    )

    for metric_name in required_keys:
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
        "Metrics.evaluate() returned an incorrect accuracy."
    )

    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-Score : {results['f1_score']:.4f}")

    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics()
