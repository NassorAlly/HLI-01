"""
preprocessing_pipeline.py

Unified preprocessing pipeline for HLI-01 landmark sequences.
"""

import numpy as np

from .smoother import LandmarkSmoother
from .normalizer import LandmarkNormalizer
from .scaler import LandmarkScaler


class PreprocessingPipeline:
    """
    Apply the HLI-01 preprocessing stages in sequence.

    Pipeline:
        smoothing
        -> normalization
        -> scaling
    """

    def __init__(
        self,
        smoothing=True,
        normalization=True,
        scaling=True,
        smoothing_window=3,
        reference_landmark=0,
    ):
        """
        Parameters
        ----------
        smoothing : bool
            Enable temporal smoothing.

        normalization : bool
            Enable landmark translation normalization.

        scaling : bool
            Enable landmark scaling.

        smoothing_window : int
            Moving-average window used by LandmarkSmoother.

        reference_landmark : int
            Landmark used as the origin by LandmarkNormalizer.
        """

        self.smoothing = smoothing
        self.normalization = normalization
        self.scaling = scaling

        self.smoother = LandmarkSmoother(
            window_size=smoothing_window
        )

        self.normalizer = LandmarkNormalizer(
            reference_landmark=reference_landmark
        )

        self.scaler = LandmarkScaler()

    def process(self, sequence):
        """
        Process one gesture sequence.

        Parameters
        ----------
        sequence : numpy.ndarray
            Expected shape (sequence_length, 63).

        Returns
        -------
        numpy.ndarray
            Processed sequence with the same shape.
        """

        sequence = np.asarray(
            sequence,
            dtype=np.float32,
        )

        if sequence.ndim != 2:
            raise ValueError(
                "Sequence must be a 2-dimensional array."
            )

        if sequence.shape[1] != 63:
            raise ValueError(
                f"Expected 63 features, got {sequence.shape[1]}"
            )

        processed = sequence.copy()

        if self.smoothing:
            processed = self.smoother(processed)

        if self.normalization:
            processed = self.normalizer(processed)

        if self.scaling:
            processed = self.scaler(processed)

        return processed.astype(np.float32)

    def process_batch(self, X):
        """
        Process a batch of gesture sequences.

        Parameters
        ----------
        X : numpy.ndarray
            Expected shape (N, sequence_length, 63).

        Returns
        -------
        numpy.ndarray
            Processed batch with the same shape.
        """

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim != 3:
            raise ValueError(
                "Batch must be a 3-dimensional array."
            )

        if X.shape[2] != 63:
            raise ValueError(
                f"Expected 63 features, got {X.shape[2]}"
            )

        processed = np.stack(
            [
                self.process(sequence)
                for sequence in X
            ]
        )

        return processed.astype(np.float32)

    def __call__(self, sequence):
        """
        Allow the pipeline to be called like a function.
        """

        return self.process(sequence)