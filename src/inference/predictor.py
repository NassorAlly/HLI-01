"""
predictor.py

Inference utilities for HLI-01.

Provides reusable prediction logic for trained models.
"""

import numpy as np
import torch

from src.config.settings import (
    DEVICE,
    NUM_FEATURES,
    SEQUENCE_LENGTH,
)


class Predictor:
    """
    Runs inference on a trained HLI-01 model.
    """

    def __init__(
        self,
        model,
        class_names,
        device=DEVICE,
    ):
        self.model = model
        self.device = torch.device(device)
        self.class_names = sorted(class_names)

        self.model.to(self.device)
        self.model.eval()

    def _validate_sequence(self, sequence):
        """
        Validate inference input shape and values.
        """

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        expected_shape = (
            SEQUENCE_LENGTH,
            NUM_FEATURES,
        )

        if sequence.shape != expected_shape:
            raise ValueError(
                f"Expected sequence shape "
                f"{expected_shape}, "
                f"got {sequence.shape}"
            )

        if np.isnan(sequence).any():
            raise ValueError(
                "Sequence contains NaN values."
            )

        if np.isinf(sequence).any():
            raise ValueError(
                "Sequence contains infinite values."
            )

        return sequence

    def predict(self, sequence):
        """
        Predict a sign from one sequence.

        Parameters
        ----------
        sequence : array-like
            Sequence with shape (30, 63).

        Returns
        -------
        dict
            Prediction information containing:
            - class_id
            - label
            - confidence
            - probabilities
        """

        sequence = self._validate_sequence(
            sequence
        )

        tensor = torch.from_numpy(
            sequence
        ).unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():
            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

        confidence, class_id = torch.max(
            probabilities,
            dim=1,
        )

        class_id = int(
            class_id.item()
        )

        confidence = float(
            confidence.item()
        )

        probabilities = probabilities.squeeze(
            0
        ).cpu().numpy()

        label = self.class_names[
            class_id
        ]

        return {
            "class_id": class_id,
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
        }