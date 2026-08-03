"""
smoother.py

Temporal smoothing utilities for HLI-01 landmark sequences.
"""

import numpy as np


class LandmarkSmoother:
    """
    Apply moving-average smoothing across frames in a
    hand-landmark sequence.
    """

    def __init__(self, window_size=3):
        """
        Parameters
        ----------
        window_size : int
            Number of frames used in the moving average.
        """
        if window_size < 1:
            raise ValueError("window_size must be at least 1.")

        self.window_size = window_size

    def smooth_sequence(self, sequence):
        """
        Smooth a complete landmark sequence.

        Parameters
        ----------
        sequence : numpy.ndarray
            Shape (sequence_length, 63)

        Returns
        -------
        numpy.ndarray
            Smoothed sequence with the same shape.
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

        if self.window_size == 1:
            return sequence.copy()

        smoothed = np.empty_like(sequence)

        for i in range(len(sequence)):
            start = max(
                0,
                i - self.window_size + 1,
            )

            smoothed[i] = np.mean(
                sequence[start:i + 1],
                axis=0,
            )

        return smoothed.astype(np.float32)

    def __call__(self, sequence):
        """
        Allow the smoother to be called like a function.
        """

        return self.smooth_sequence(sequence)