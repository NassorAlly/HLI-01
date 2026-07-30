"""
=========================================================
HLI-01
Evaluation Metrics
=========================================================
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class Metrics:
    """Computes classification metrics."""

    @staticmethod
    def accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def precision(y_true, y_pred):
        return precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

    @staticmethod
    def recall(y_true, y_pred):
        return recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

    @staticmethod
    def f1(y_true, y_pred):
        return f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

    @staticmethod
    def evaluate(y_true, y_pred):
        return {
            "accuracy": Metrics.accuracy(y_true, y_pred),
            "precision": Metrics.precision(y_true, y_pred),
            "recall": Metrics.recall(y_true, y_pred),
            "f1_score": Metrics.f1(y_true, y_pred),
        }
