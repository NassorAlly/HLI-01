"""
=========================================================
HLI-01
Confusion Matrix
Version 0.4.0
=========================================================
Computes the confusion matrix for classification.
"""

from sklearn.metrics import confusion_matrix


class ConfusionMatrix:
    """
    Computes the confusion matrix.
    """

    @staticmethod
    def compute(y_true, y_pred):
        """
        Compute confusion matrix.

        Parameters
        ----------
        y_true : list or ndarray
            Ground truth labels.

        y_pred : list or ndarray
            Predicted labels.

        Returns
        -------
        ndarray
            Confusion matrix.
        """
        return confusion_matrix(y_true, y_pred)
