from typing import Dict, List, Any

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


class ResearchMetrics:
    """
    Compute research-ready classification metrics for HLI-01.
    """

    @staticmethod
    def compute(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str],
    ) -> Dict[str, Any]:
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        if len(y_true) == 0:
            raise ValueError("Prediction data cannot be empty.")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_weighted": precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "recall_weighted": recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "f1_weighted": f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            ),
            "precision_macro": precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "recall_macro": recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "f1_macro": f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            "classification_report": classification_report(
                y_true,
                y_pred,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_true,
                y_pred,
            ).tolist(),
        }

        return metrics