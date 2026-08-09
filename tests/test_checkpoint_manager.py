import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.lstm_model import LSTMModel
from src.training.checkpoint_manager import CheckpointManager


def create_model():
    return LSTMModel(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
    )


def test_checkpoint_manager():
    with tempfile.TemporaryDirectory() as temp_dir:

        model = create_model()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
        )

        manager = CheckpointManager(
            checkpoint_dir=temp_dir
        )

        manager.save(
            model=model,
            optimizer=optimizer,
            epoch=5,
            loss=0.1234,
        )

        epoch, loss = manager.load(
            model=model,
            optimizer=optimizer,
        )

        assert epoch == 5
        assert abs(loss - 0.1234) < 1e-6


def test_model_only_checkpoint_loading():
    with tempfile.TemporaryDirectory() as temp_dir:

        original_model = create_model()

        optimizer = torch.optim.Adam(
            original_model.parameters(),
            lr=0.001,
        )

        manager = CheckpointManager(
            checkpoint_dir=temp_dir
        )

        manager.save(
            model=original_model,
            optimizer=optimizer,
            epoch=5,
            loss=0.1234,
        )

        inference_model = create_model()

        checkpoint = manager.load_model(
            model=inference_model,
        )

        assert checkpoint["epoch"] == 5
        assert abs(
            checkpoint["loss"] - 0.1234
        ) < 1e-6

        for original_param, loaded_param in zip(
            original_model.parameters(),
            inference_model.parameters(),
        ):
            assert torch.equal(
                original_param,
                loaded_param,
            )