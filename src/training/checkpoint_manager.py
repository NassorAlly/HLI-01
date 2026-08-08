"""
checkpoint_manager.py

Saves and loads model checkpoints for HLI-01.
"""

from pathlib import Path

import torch


class CheckpointManager:
    """
    Handles model checkpoint saving and loading.
    """

    def __init__(
        self,
        checkpoint_dir="outputs/checkpoints",
    ):

        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model,
        optimizer,
        epoch,
        loss,
        filename="best_model.pth",
    ):

        filepath = self.checkpoint_dir / filename

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            filepath,
        )

        return filepath

    def load(
        self,
        model,
        optimizer,
        filename="best_model.pth",
        device="cpu",
    ):

        filepath = self.checkpoint_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {filepath}"
            )

        checkpoint = torch.load(
            filepath,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        return epoch, loss