"""
train.py

Main training entry point for HLI-01 v0.8.0.

Uses the centralized project configuration to run
the complete model-training pipeline.
"""

import random

import numpy as np
import torch

from src.config.settings import (
    NUM_CLASSES,
    NUM_EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    INPUT_SIZE,
    HIDDEN_SIZE,
    NUM_LAYERS,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    FIGURE_DIR,
)

from src.models.lstm_model import LSTMModel
from src.training.data_loader import DataLoaderManager
from src.training.trainer import Trainer
from src.visualization import TrainingPlotter


def set_random_seed(seed):
    """
    Configure random seeds for reproducible training.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)


def main():
    """
    Run the complete HLI-01 v0.8.0 training pipeline.
    """

    print("=" * 60)
    print("HLI-01 v0.8.0 - Training Pipeline")
    print("=" * 60)

    # --------------------------------------------------
    # Reproducibility
    # --------------------------------------------------

    set_random_seed(
        RANDOM_SEED
    )

    print()
    print(
        "Random seed       :",
        RANDOM_SEED,
    )

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device            :",
        device,
    )

    # --------------------------------------------------
    # Configuration summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Training Configuration")
    print("=" * 60)

    print(
        "Input size        :",
        INPUT_SIZE,
    )

    print(
        "Hidden size       :",
        HIDDEN_SIZE,
    )

    print(
        "LSTM layers       :",
        NUM_LAYERS,
    )

    print(
        "Number of classes :",
        NUM_CLASSES,
    )

    print(
        "Maximum epochs    :",
        NUM_EPOCHS,
    )

    print(
        "Learning rate     :",
        LEARNING_RATE,
    )

    print(
        "Scheduler factor  :",
        LR_SCHEDULER_FACTOR,
    )

    print(
        "Scheduler patience:",
        LR_SCHEDULER_PATIENCE,
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Loading Dataset")
    print("=" * 60)

    data_manager = (
        DataLoaderManager()
    )

    (
        train_loader,
        valid_loader,
        test_loader,
    ) = data_manager.create()

    print()
    print(
        "Training batches  :",
        len(train_loader),
    )

    print(
        "Validation batches:",
        len(valid_loader),
    )

    print(
        "Testing batches   :",
        len(test_loader),
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Creating Model")
    print("=" * 60)

    model = LSTMModel(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
    )

    model = model.to(
        device
    )

    print()
    print(
        "Model created successfully."
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
        )
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
        epochs=NUM_EPOCHS,
        scheduler=scheduler,
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Starting Training")
    print("=" * 60)

    history = trainer.train()

    # --------------------------------------------------
    # Training summary
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Training Complete")
    print("=" * 60)

    print()

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
        len(
            history["train_loss"]
        ),
    )

    final_lr = (
        optimizer.param_groups[0]["lr"]
    )

    print(
        "Final learning rate :",
        f"{final_lr:.6f}",
    )

    # --------------------------------------------------
    # Training visualizations
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Generating Training Visualizations")
    print("=" * 60)

    plotter = TrainingPlotter(
        save_dir=FIGURE_DIR
    )

    plot_outputs = (
        plotter.plot_history(
            history
        )
    )

    print()

    if "loss" in plot_outputs:

        print(
            "Loss curve         :",
            plot_outputs["loss"],
        )

    if "accuracy" in plot_outputs:

        print(
            "Accuracy curve     :",
            plot_outputs["accuracy"],
        )

    if "learning_rate" in plot_outputs:

        print(
            "Learning-rate curve:",
            plot_outputs[
                "learning_rate"
            ],
        )

    # --------------------------------------------------
    # Outputs
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("Training Outputs")
    print("=" * 60)

    print()

    print(
        "Best checkpoint:"
    )

    print(
        "outputs/checkpoints/best_model.pth"
    )

    print()

    print(
        "Training figures:"
    )

    for output_path in (
        plot_outputs.values()
    ):

        print(
            output_path
        )

    # --------------------------------------------------
    # Final status
    # --------------------------------------------------

    print()
    print("=" * 60)

    print(
        "HLI-01 v0.8.0 training "
        "pipeline completed successfully."
    )

    print("=" * 60)


if __name__ == "__main__":
    main()