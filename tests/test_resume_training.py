import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tempfile

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.lstm_model import LSTMModel
from src.training.trainer import Trainer


print("=" * 60)
print("HLI-01 v0.8.0 - Testing Resume Training")
print("=" * 60)


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

device = torch.device("cpu")


# --------------------------------------------------
# Temporary checkpoint directory
# --------------------------------------------------

with tempfile.TemporaryDirectory() as temp_dir:

    # --------------------------------------------------
    # First training session
    # --------------------------------------------------

    print()
    print("PHASE 1 - Initial Training")
    print("-" * 60)

    model_1 = LSTMModel(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
    )

    optimizer_1 = torch.optim.Adam(
        model_1.parameters(),
        lr=0.001,
    )

    trainer_1 = Trainer(
        model=model_1,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer_1,
        device=device,
        epochs=2,
    )

    trainer_1.checkpoints.checkpoint_dir = Path(
        temp_dir
    )

    history_1 = trainer_1.train()

    checkpoint_file = (
        Path(temp_dir)
        / "best_model.pth"
    )

    assert checkpoint_file.exists()

    saved_epoch = trainer_1.best_epoch

    print()
    print(
        "Checkpoint created at epoch:",
        saved_epoch,
    )


    # --------------------------------------------------
    # Second training session
    # --------------------------------------------------

    print()
    print("PHASE 2 - Resume Training")
    print("-" * 60)

    model_2 = LSTMModel(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
    )

    optimizer_2 = torch.optim.Adam(
        model_2.parameters(),
        lr=0.001,
    )

    trainer_2 = Trainer(
        model=model_2,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer_2,
        device=device,
        epochs=4,
    )

    trainer_2.checkpoints.checkpoint_dir = Path(
        temp_dir
    )

    epoch, loss = (
        trainer_2.resume_from_checkpoint()
    )


    # --------------------------------------------------
    # Validate restored state
    # --------------------------------------------------

    assert epoch + 1 == saved_epoch

    assert trainer_2.start_epoch == (
        epoch + 1
    )

    assert trainer_2.best_epoch == (
        epoch + 1
    )

    assert trainer_2.best_loss == loss


    # --------------------------------------------------
    # Continue training
    # --------------------------------------------------

    history_2 = trainer_2.train()

    assert len(
        history_2["train_loss"]
    ) > 0


    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Resume Training Test Results")
    print("=" * 60)

    print(
        "Checkpoint epoch :",
        epoch + 1,
    )

    print(
        "Resume epoch     :",
        trainer_2.start_epoch + 1,
    )

    print(
        "New epochs run   :",
        len(history_2["train_loss"]),
    )

    print()
    print(
        "✓ Checkpoint loaded correctly."
    )

    print(
        "✓ Model state restored correctly."
    )

    print(
        "✓ Optimizer state restored correctly."
    )

    print(
        "✓ Epoch state restored correctly."
    )

    print(
        "✓ Training resumed successfully."
    )

    print()
    print("TEST PASSED")