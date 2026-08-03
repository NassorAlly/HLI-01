"""
scaler.py

Feature scaling utilities for HLI-01 landmark sequences.
"""

import numpy as np


class LandmarkScaler:
    """
    Scale hand-landmark coordinates to reduce variation
    caused by different hand sizes and camera distances.
    """

    def __init__(self, epsilon=1e-8):
        """
        Parameters
        ----------
        epsilon : float
            Small constant used to avoid division by zero.
        """
        self.epsilon = epsilon

    def scale_frame(self, frame):
        """
        Scale one frame using the maximum Euclidean distance
        of any landmark from the origin.

        Parameters
        ----------
        frame : numpy.ndarray
            Shape (63,)

        Returns
        -------
        numpy.ndarray
            Scaled frame with shape (63,)
        """

        frame = np.asarray(frame, dtype=np.float32)

        if frame.shape != (63,):
            raise ValueError(
                f"Expected frame shape (63,), got {frame.shape}"
            )

        landmarks = frame.reshape(21, 3)

        distances = np.linalg.norm(
            landmarks,
            axis=1,
        )

        scale = np.max(distances)

        if scale < self.epsilon:
            return frame.copy()

        scaled = landmarks / scale

        return scaled.reshape(63).astype(np.float32)

    def scale_sequence(self, sequence):
        """
        Scale an entire gesture sequence.

        Parameters
        ----------
        sequence : numpy.ndarray
            Shape (sequence_length, 63)

        Returns
        -------
        numpy.ndarray
            Scaled sequence with the same shape.
        """

        sequence = np.asarray(sequence, dtype=np.float32)

        if sequence.ndim != 2:
            raise ValueError(
                "Sequence must be a 2-dimensional array."
            )

        if sequence.shape[1] != 63:
            raise ValueError(
                f"Expected 63 features, got {sequence.shape[1]}"
            )

        scaled = np.stack(
            [
                self.scale_frame(frame)
                for frame in sequence
            ]
        )

        return scaled.astype(np.float32)

    def __call__(self, sequence):
        """
        Allow the scaler to be called like a function.
        """

        return self.scale_sequence(sequence)