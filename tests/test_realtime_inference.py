import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.config.settings import (
    HIDDEN_SIZE,
    INPUT_SIZE,
    NUM_CLASSES,
    NUM_LAYERS,
    NUM_FEATURES,
)
from src.inference.realtime_inference import (
    create_model,
    extract_keypoints,
)


class DummyLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class DummyHandLandmarks:
    def __init__(self):
        self.landmark = [
            DummyLandmark(
                x=i / 100.0,
                y=i / 200.0,
                z=i / 300.0,
            )
            for i in range(21)
        ]


class DummyResults:
    def __init__(self, detected=True):
        if detected:
            self.multi_hand_landmarks = [
                DummyHandLandmarks()
            ]
        else:
            self.multi_hand_landmarks = None


def test_create_model_returns_correct_architecture():

    model = create_model(
        device=torch.device("cpu")
    )

    assert model.lstm.input_size == INPUT_SIZE
    assert model.lstm.hidden_size == HIDDEN_SIZE
    assert model.lstm.num_layers == NUM_LAYERS

    assert (
        model.classifier.out_features
        == NUM_CLASSES
    )


def test_create_model_moves_to_requested_device():

    model = create_model(
        device=torch.device("cpu")
    )

    parameter = next(
        model.parameters()
    )

    assert parameter.device.type == "cpu"


def test_extract_keypoints_returns_correct_shape():

    results = DummyResults(
        detected=True
    )

    keypoints = extract_keypoints(
        results
    )

    assert isinstance(
        keypoints,
        np.ndarray,
    )

    assert keypoints.shape == (
        NUM_FEATURES,
    )


def test_extract_keypoints_returns_float32():

    results = DummyResults(
        detected=True
    )

    keypoints = extract_keypoints(
        results
    )

    assert keypoints.dtype == np.float32


def test_extract_keypoints_returns_none_when_no_hand():

    results = DummyResults(
        detected=False
    )

    keypoints = extract_keypoints(
        results
    )

    assert keypoints is None