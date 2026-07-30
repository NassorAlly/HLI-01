"""
trainer.py

Training engine for HLI-01.
"""

import torch
import torch.nn as nn
from tqdm import tqdm

from src.training.early_stopping import EarlyStopping
from src.training.checkpoint_manager import CheckpointManager


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        device,
        epochs=30,
    ):

        self.model = model

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.optimizer = optimizer

        self.device = device

        self.epochs = epochs

        self.criterion = nn.CrossEntropyLoss()

        self.early_stopping = EarlyStopping()

        self.checkpoints = CheckpointManager()

    def train(self):

        self.model.to(self.device)

        best_loss = float("inf")

        for epoch in range(self.epochs):

            print()

            print("=" * 60)

            print(f"Epoch {epoch+1}/{self.epochs}")

            print("=" * 60)

            self.model.train()

            running_loss = 0.0

            running_correct = 0

            running_total = 0

            for inputs, labels in tqdm(self.train_loader):

                inputs = inputs.to(self.device)

                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)

                loss = self.criterion(
                    outputs,
                    labels,
                )

                loss.backward()

                self.optimizer.step()

                running_loss += loss.item()

                predictions = torch.argmax(
                    outputs,
                    dim=1,
                )

                running_correct += (
                    predictions == labels
                ).sum().item()

                running_total += labels.size(0)

            train_loss = running_loss / len(
                self.train_loader
            )

            train_accuracy = (
                running_correct
                / running_total
            )

            validation_loss = self.validate()

            print()

            print(
                f"Train Loss     : {train_loss:.4f}"
            )

            print(
                f"Train Accuracy : {train_accuracy:.4f}"
            )

            print(
                f"Validation Loss: {validation_loss:.4f}"
            )

            if validation_loss < best_loss:

                best_loss = validation_loss

                self.checkpoints.save(
                    self.model,
                    self.optimizer,
                    epoch,
                    validation_loss,
                )

            if self.early_stopping(validation_loss):

                print()

                print("Early stopping triggered.")

                break

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        with torch.no_grad():

            for inputs, labels in self.valid_loader:

                inputs = inputs.to(self.device)

                labels = labels.to(self.device)

                outputs = self.model(inputs)

                loss = self.criterion(
                    outputs,
                    labels,
                )

                running_loss += loss.item()

        return running_loss / len(self.valid_loader)
