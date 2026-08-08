"""
trainer.py

Training engine for HLI-01.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from src.config.settings import (
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
)

from src.training.early_stopping import EarlyStopping
from src.training.checkpoint_manager import CheckpointManager


class Trainer:
    """
    Handles model training, validation,
    checkpointing, resume training,
    early stopping, and learning-rate scheduling.
    """

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        device,
        epochs=30,
        scheduler=None,
    ):

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.epochs = epochs
        self.scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss()

        self.early_stopping = EarlyStopping(
            patience=EARLY_STOPPING_PATIENCE,
            min_delta=EARLY_STOPPING_MIN_DELTA,
        )

        self.checkpoints = CheckpointManager()

        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "valid_loss": [],
            "valid_accuracy": [],
            "learning_rate": [],
        }

        self.best_loss = float("inf")
        self.best_epoch = None
        self.start_epoch = 0

    def resume_from_checkpoint(
        self,
        filename="best_model.pth",
    ):
        """
        Resume model and optimizer state
        from a saved checkpoint.
        """

        epoch, loss = self.checkpoints.load(
            model=self.model,
            optimizer=self.optimizer,
            filename=filename,
            device=self.device,
        )

        self.start_epoch = epoch + 1

        self.best_loss = loss
        self.best_epoch = epoch + 1

        self.early_stopping.best_loss = loss
        self.early_stopping.counter = 0
        self.early_stopping.stop = False

        print()
        print("=" * 60)
        print("Checkpoint Loaded")
        print("=" * 60)

        print(
            f"Saved epoch       : {epoch + 1}"
        )

        print(
            f"Validation loss   : {loss:.4f}"
        )

        print(
            f"Resume from epoch : {self.start_epoch + 1}"
        )

        return epoch, loss

    def train(self):
        """
        Train the model.

        Returns
        -------
        dict
            Training history.
        """

        self.model.to(self.device)

        for epoch in range(
            self.start_epoch,
            self.epochs,
        ):

            print()
            print("=" * 60)
            print(
                f"Epoch {epoch + 1}/{self.epochs}"
            )
            print("=" * 60)

            train_loss, train_accuracy = (
                self._train_epoch()
            )

            valid_loss, valid_accuracy = (
                self.validate()
            )

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            self.history[
                "train_loss"
            ].append(train_loss)

            self.history[
                "train_accuracy"
            ].append(train_accuracy)

            self.history[
                "valid_loss"
            ].append(valid_loss)

            self.history[
                "valid_accuracy"
            ].append(valid_accuracy)

            self.history[
                "learning_rate"
            ].append(current_lr)

            print()

            print(
                f"Train Loss      : "
                f"{train_loss:.4f}"
            )

            print(
                f"Train Accuracy  : "
                f"{train_accuracy:.4f}"
            )

            print(
                f"Validation Loss : "
                f"{valid_loss:.4f}"
            )

            print(
                f"Validation Acc. : "
                f"{valid_accuracy:.4f}"
            )

            print(
                f"Learning Rate   : "
                f"{current_lr:.6f}"
            )

            if valid_loss < self.best_loss:

                self.best_loss = valid_loss
                self.best_epoch = epoch + 1

                self.checkpoints.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    valid_loss,
                )

                print(
                    f"Best model saved at "
                    f"epoch {self.best_epoch}"
                )

            if self.scheduler is not None:

                self.scheduler.step(
                    valid_loss
                )

            if self.early_stopping(
                valid_loss
            ):

                print()
                print(
                    "Early stopping triggered."
                )

                break

        return self.history

    def _train_epoch(self):
        """
        Run one training epoch.
        """

        self.model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for inputs, labels in tqdm(
            self.train_loader,
            desc="Training",
            leave=False,
        ):

            inputs = inputs.to(
                self.device
            )

            labels = labels.to(
                self.device
            )

            self.optimizer.zero_grad()

            outputs = self.model(
                inputs
            )

            loss = self.criterion(
                outputs,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            running_loss += (
                loss.item()
            )

            predictions = torch.argmax(
                outputs,
                dim=1,
            )

            running_correct += (
                predictions == labels
            ).sum().item()

            running_total += (
                labels.size(0)
            )

        train_loss = (
            running_loss
            / len(self.train_loader)
        )

        train_accuracy = (
            running_correct
            / running_total
        )

        return (
            train_loss,
            train_accuracy,
        )

    def validate(self):
        """
        Evaluate the model on the validation dataset.
        """

        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with torch.no_grad():

            for inputs, labels in (
                self.valid_loader
            ):

                inputs = inputs.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                outputs = self.model(
                    inputs
                )

                loss = self.criterion(
                    outputs,
                    labels,
                )

                running_loss += (
                    loss.item()
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

                running_correct += (
                    predictions == labels
                ).sum().item()

                running_total += (
                    labels.size(0)
                )

        valid_loss = (
            running_loss
            / len(self.valid_loader)
        )

        valid_accuracy = (
            running_correct
            / running_total
        )

        return (
            valid_loss,
            valid_accuracy,
        )