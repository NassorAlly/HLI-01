"""
normalizer.py

Normalization utilities for HLI-01 hand-landmark sequences.
"""

import numpy as np


class LandmarkNormalizer:
    """
    Normalize MediaPipe hand-landmark coordinates.

    Each HLI-01 frame contains 21 landmarks with
    three coordinates per landmark:

        21 × 3 = 63 features
    """

    def __init__(self, reference_landmark=0):
        """
        Parameters
        ----------
        reference_landmark : int
            Landmark used as the coordinate origin.
            MediaPipe landmark 0 corresponds to the wrist.
        """

        self.reference_landmark = reference_landmark

    def normalize_frame(self, frame):
        """
        Normalize one frame relative to the reference landmark.

        Parameters
        ----------
        frame : numpy.ndarray
            Shape (63,)

        Returns
        -------
        numpy.ndarray
            Normalized frame with shape (63,)
        """

        frame = np.asarray(frame, dtype=np.float32)

        if frame.shape != (63,):
            raise ValueError(
                f"Expected frame shape (63,), got {frame.shape}"
            )

        landmarks = frame.reshape(21, 3)

        reference = landmarks[self.reference_landmark].copy()

        normalized = landmarks - reference

        return normalized.reshape(63).astype(np.float32)

    def normalize_sequence(self, sequence):
        """
        Normalize a complete gesture sequence.

        Parameters
        ----------
        sequence : numpy.ndarray
            Shape (sequence_length, 63)

        Returns
        -------
        numpy.ndarray
            Normalized sequence with the same shape.
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

        normalized = np.stack(
            [
                self.normalize_frame(frame)
                for frame in sequence
            ]
        )

        return normalized.astype(np.float32)

    def __call__(self, sequence):
        """
        Allow the normalizer to be called like a function.
        """

        return self.normalize_sequence(sequence)