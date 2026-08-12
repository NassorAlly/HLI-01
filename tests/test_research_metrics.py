import pytest

from src.evaluation.research_metrics import ResearchMetrics


def test_compute_research_metrics():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]

    class_names = [
        "hello",
        "no",
        "peace",
        "yes",
    ]

    metrics = ResearchMetrics.compute(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_weighted"] == 1.0
    assert metrics["recall_weighted"] == 1.0
    assert metrics["f1_weighted"] == 1.0

    assert metrics["precision_macro"] == 1.0
    assert metrics["recall_macro"] == 1.0
    assert metrics["f1_macro"] == 1.0

    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics

    assert len(metrics["confusion_matrix"]) == 4


def test_mismatched_lengths_raise_error():
    with pytest.raises(ValueError):
        ResearchMetrics.compute(
            y_true=[0, 1],
            y_pred=[0],
            class_names=["hello", "no"],
        )


def test_empty_predictions_raise_error():
    with pytest.raises(ValueError):
        ResearchMetrics.compute(
            y_true=[],
            y_pred=[],
            class_names=[],
        )