"""
=========================================================
HLI-01
Evaluator
Version 0.4.0
=========================================================
Combines evaluation metrics and confusion matrix.
"""

from src.evaluation.metrics import Metrics
from src.evaluation.confusion_matrix import ConfusionMatrix


class Evaluator:
    """
    Performs complete evaluation of classification results.
    """

    @staticmethod
    def evaluate(y_true, y_pred):
        """
        Evaluate predictions.

        Parameters
        ----------
        y_true : list or ndarray
            Ground truth labels.

        y_pred : list or ndarray
            Predicted labels.

        Returns
        -------
        dict
            Dictionary containing all evaluation results.
        """

        results = Metrics.evaluate(y_true, y_pred)

        results["confusion_matrix"] = (
            ConfusionMatrix.compute(y_true, y_pred)
        )

        return results
