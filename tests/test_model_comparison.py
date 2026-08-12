import pytest

from src.evaluation.model_comparison import ModelComparison


def test_compare_models_by_accuracy():
    experiments = [
        {
            "name": "BiLSTM",
            "accuracy": 0.95,
        },
        {
            "name": "BiLSTM_Attention",
            "accuracy": 0.98,
        },
        {
            "name": "LSTM",
            "accuracy": 0.92,
        },
    ]

    results = ModelComparison.compare(
        experiments,
        metric="accuracy",
    )

    assert results[0]["name"] == "BiLSTM_Attention"
    assert results[1]["name"] == "BiLSTM"
    assert results[2]["name"] == "LSTM"


def test_compare_models_by_f1():
    experiments = [
        {
            "name": "Model_A",
            "f1_macro": 0.90,
        },
        {
            "name": "Model_B",
            "f1_macro": 0.95,
        },
    ]

    results = ModelComparison.compare(
        experiments,
        metric="f1_macro",
    )

    assert results[0]["name"] == "Model_B"


def test_empty_experiments_raise_error():
    with pytest.raises(ValueError):
        ModelComparison.compare([])


def test_missing_metric_raises_error():
    experiments = [
        {
            "name": "Model_A",
            "accuracy": 0.90,
        }
    ]

    with pytest.raises(KeyError):
        ModelComparison.compare(
            experiments,
            metric="f1_macro",
        )