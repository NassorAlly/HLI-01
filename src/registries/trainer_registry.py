"""
train.py

Main training entry point for HLI-01.
"""

import torch

from src.models.lstm_model import LSTMModel
from src.training.data_loader import DataLoaderManager
from src.training.trainer import Trainer


def main():
    print("=" * 60)
    print("HLI-01 v0.8.0 - Training Pipeline")
    print("=" * 60)

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Device:", device)

    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    print()
    print("Loading dataset...")

    data_manager = DataLoaderManager()

    (
        train_loader,
        valid_loader,
        test_loader,
    ) = data_manager.create()

    print("Training batches  :", len(train_loader))
    print("Validation batches:", len(valid_loader))
    print("Testing batches   :", len(test_loader))

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print()
    print("Creating model...")

    model = LSTMModel(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
    )

    model = model.to(device)

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
        device=device,
        epochs=30,
    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    print()
    print("Starting training...")

    history = trainer.train()

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)

    print(
        "Best epoch          :",
        trainer.best_epoch,
    )

    print(
        "Best validation loss:",
        f"{trainer.best_loss:.4f}",
    )

    print(
        "Epochs completed    :",
        len(history["train_loss"]),
    )

    print()
    print(
        "Best checkpoint saved to:"
    )

    print(
        "outputs/checkpoints/best_model.pth"
    )


if __name__ == "__main__":
    main()