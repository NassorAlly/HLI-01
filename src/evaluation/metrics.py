"""
=========================================================
HLI-01
Evaluation Metrics
Version 0.4.0
=========================================================
Computes classification performance metrics.
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Metrics:
    """
    Computes classification metrics for sign language recognition.
    """

    @staticmethod
    def accuracy(y_true, y_pred):
        """Compute classification accuracy."""
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        """Compute weighted precision."""
        return precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

    @staticmethod
    def recall(y_true, y_pred):
        """Compute weighted recall."""
        return recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

    @staticmethod
    def f1(y_true, y_pred):
        """Compute weighted F1-score."""
        return f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0
        )

    @staticmethod
    def evaluate(y_true, y_pred):
        """Return all evaluation metrics."""
        return {
            "accuracy": Metrics.accuracy(y_true, y_pred),
            "precision": Metrics.precision(y_true, y_pred),
            "recall": Metrics.recall(y_true, y_pred),
            "f1_score": Metrics.f1(y_true, y_pred),
        }
