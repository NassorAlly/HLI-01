import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest
import torch

from src.inference.predictor import Predictor
from src.models.lstm_model import LSTMModel


CLASS_NAMES = [
    "hello",
    "no",
    "peace",
    "yes",
]


def create_model():
    return LSTMModel(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
    )


def create_predictor():
    model = create_model()

    return Predictor(
        model=model,
        class_names=CLASS_NAMES,
        device="cpu",
    )


def test_predictor_returns_expected_keys():
    predictor = create_predictor()

    sequence = np.random.rand(
        30,
        63,
    ).astype(np.float32)

    result = predictor.predict(sequence)

    assert "class_id" in result
    assert "label" in result
    assert "confidence" in result
    assert "probabilities" in result


def test_predictor_returns_valid_class():
    predictor = create_predictor()

    sequence = np.random.rand(
        30,
        63,
    ).astype(np.float32)

    result = predictor.predict(sequence)

    assert result["class_id"] in [
        0,
        1,
        2,
        3,
    ]

    assert result["label"] in CLASS_NAMES


def test_predictor_confidence_range():
    predictor = create_predictor()

    sequence = np.random.rand(
        30,
        63,
    ).astype(np.float32)

    result = predictor.predict(sequence)

    assert 0.0 <= result["confidence"] <= 1.0


def test_predictor_probabilities_sum_to_one():
    predictor = create_predictor()

    sequence = np.random.rand(
        30,
        63,
    ).astype(np.float32)

    result = predictor.predict(sequence)

    probabilities = result["probabilities"]

    assert probabilities.shape == (4,)

    assert np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-6,
    )


def test_predictor_rejects_invalid_shape():
    predictor = create_predictor()

    sequence = np.random.rand(
        29,
        63,
    ).astype(np.float32)

    with pytest.raises(ValueError):
        predictor.predict(sequence)


def test_predictor_rejects_nan_values():
    predictor = create_predictor()

    sequence = np.random.rand(
        30,
        63,
    ).astype(np.float32)

    sequence[0, 0] = np.nan

    with pytest.raises(ValueError):
        predictor.predict(sequence)


def test_predictor_rejects_infinite_values():
    predictor = create_predictor()

    sequence = np.random.rand(
        30,
        63,
    ).astype(np.float32)

    sequence[0, 0] = np.inf

    with pytest.raises(ValueError):
        predictor.predict(sequence)


def test_predictor_sets_model_to_eval_mode():
    model = create_model()

    model.train()

    Predictor(
        model=model,
        class_names=CLASS_NAMES,
        device="cpu",
    )

    assert model.training is False


def test_predictor_does_not_change_model_parameters():
    model = create_model()

    predictor = Predictor(
        model=model,
        class_names=CLASS_NAMES,
        device="cpu",
    )

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    sequence = np.random.rand(
        30,
        63,
    ).astype(np.float32)

    predictor.predict(sequence)

    after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    for before_parameter, after_parameter in zip(
        before,
        after,
    ):
        assert torch.equal(
            before_parameter,
            after_parameter,
        )