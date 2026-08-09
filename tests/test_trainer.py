import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_model import LSTMModel
from src.training.trainer import Trainer


def test_trainer(tmp_path):
    """
    Smoke-test the training pipeline for two epochs.

    The checkpoint is written only to pytest's
    temporary directory so the real trained model
    can never be overwritten.
    """

    # --------------------------------------------------
    # Synthetic dataset
    # --------------------------------------------------

    X = torch.randn(
        64,
        30,
        63,
    )

    y = torch.randint(
        0,
        4,
        (64,),
    )

    dataset = TensorDataset(
        X,
        y,
    )

    # --------------------------------------------------
    # Data loaders
    # --------------------------------------------------

    train_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
    )

    valid_loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = LSTMModel(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        device=torch.device("cpu"),
        epochs=2,
    )

    # IMPORTANT:
    # Redirect smoke-test checkpoints away from
    # outputs/checkpoints/best_model.pth.

    trainer.checkpoints.checkpoint_dir = Path(
        tmp_path
    )

    # --------------------------------------------------
    # Run two-epoch smoke training
    # --------------------------------------------------

    history = trainer.train()

    # --------------------------------------------------
    # Validate training history
    # --------------------------------------------------

    assert isinstance(
        history,
        dict,
    )

    assert "train_loss" in history
    assert "train_accuracy" in history
    assert "valid_loss" in history
    assert "valid_accuracy" in history

    assert len(
        history["train_loss"]
    ) == 2

    assert len(
        history["train_accuracy"]
    ) == 2

    assert len(
        history["valid_loss"]
    ) == 2

    assert len(
        history["valid_accuracy"]
    ) == 2

    # --------------------------------------------------
    # Validate metrics
    # --------------------------------------------------

    for loss in history[
        "train_loss"
    ]:
        assert loss >= 0.0

    for loss in history[
        "valid_loss"
    ]:
        assert loss >= 0.0

    for accuracy in history[
        "train_accuracy"
    ]:
        assert (
            0.0
            <= accuracy
            <= 1.0
        )

    for accuracy in history[
        "valid_accuracy"
    ]:
        assert (
            0.0
            <= accuracy
            <= 1.0
        )

    # --------------------------------------------------
    # Validate checkpoint tracking
    # --------------------------------------------------

    assert trainer.best_epoch is not None

    assert trainer.best_loss < float(
        "inf"
    )

    checkpoint_file = (
        Path(tmp_path)
        / "best_model.pth"
    )

    assert checkpoint_file.exists()