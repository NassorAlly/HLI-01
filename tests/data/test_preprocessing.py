"""
test_preprocessing.py

Unit tests for the HLI-01 preprocessing pipeline.
"""

import numpy as np
import pytest

from src.preprocessing.normalizer import LandmarkNormalizer
from src.preprocessing.scaler import LandmarkScaler
from src.preprocessing.smoother import LandmarkSmoother
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline


def create_sequence():
    """
    Create a synthetic HLI-01 sequence.

    Shape:
        30 frames
        63 features per frame
    """
    return np.random.rand(
        30,
        63,
    ).astype(np.float32)


def test_normalizer_preserves_shape():
    """
    Test that normalization preserves sequence dimensions.
    """
    sequence = create_sequence()

    normalizer = LandmarkNormalizer()
    result = normalizer(sequence)

    assert result.shape == (30, 63)
    assert result.dtype == np.float32


def test_normalizer_sets_reference_to_zero():
    """
    Test that the wrist landmark becomes the origin.
    """
    sequence = create_sequence()

    normalizer = LandmarkNormalizer(
        reference_landmark=0
    )

    result = normalizer(sequence)

    landmarks = result.reshape(
        30,
        21,
        3,
    )

    np.testing.assert_allclose(
        landmarks[:, 0, :],
        0.0,
        atol=1e-6,
    )


def test_scaler_preserves_shape():
    """
    Test that scaling preserves sequence dimensions.
    """
    sequence = create_sequence()

    scaler = LandmarkScaler()
    result = scaler(sequence)

    assert result.shape == (30, 63)
    assert result.dtype == np.float32


def test_scaler_limits_max_distance():
    """
    Test that scaled landmarks do not exceed
    unit distance from the origin.
    """
    sequence = create_sequence()

    normalizer = LandmarkNormalizer()
    scaler = LandmarkScaler()

    normalized = normalizer(sequence)
    scaled = scaler(normalized)

    landmarks = scaled.reshape(
        30,
        21,
        3,
    )

    distances = np.linalg.norm(
        landmarks,
        axis=2,
    )

    assert np.max(distances) <= 1.000001


def test_smoother_preserves_shape():
    """
    Test that temporal smoothing preserves dimensions.
    """
    sequence = create_sequence()

    smoother = LandmarkSmoother(
        window_size=3
    )

    result = smoother(sequence)

    assert result.shape == (30, 63)
    assert result.dtype == np.float32


def test_smoother_window_one_returns_same_values():
    """
    Test that window size 1 leaves the sequence unchanged.
    """
    sequence = create_sequence()

    smoother = LandmarkSmoother(
        window_size=1
    )

    result = smoother(sequence)

    np.testing.assert_allclose(
        result,
        sequence,
        rtol=1e-6,
        atol=1e-6,
    )


def test_pipeline_preserves_shape():
    """
    Test the complete preprocessing pipeline.
    """
    sequence = create_sequence()

    pipeline = PreprocessingPipeline()

    result = pipeline.process(sequence)

    assert result.shape == (30, 63)
    assert result.dtype == np.float32


def test_pipeline_process_batch():
    """
    Test preprocessing of multiple sequences.
    """
    X = np.random.rand(
        8,
        30,
        63,
    ).astype(np.float32)

    pipeline = PreprocessingPipeline()

    result = pipeline.process_batch(X)

    assert result.shape == (8, 30, 63)
    assert result.dtype == np.float32


def test_pipeline_can_disable_stages():
    """
    Test that all preprocessing stages can be disabled.
    """
    sequence = create_sequence()

    pipeline = PreprocessingPipeline(
        smoothing=False,
        normalization=False,
        scaling=False,
    )

    result = pipeline.process(sequence)

    np.testing.assert_allclose(
        result,
        sequence,
        rtol=1e-6,
        atol=1e-6,
    )


def test_pipeline_rejects_invalid_feature_count():
    """
    Test rejection of sequences with incorrect feature count.
    """
    sequence = np.random.rand(
        30,
        60,
    ).astype(np.float32)

    pipeline = PreprocessingPipeline()

    with pytest.raises(ValueError):
        pipeline.process(sequence)


def test_pipeline_rejects_invalid_batch_dimensions():
    """
    Test rejection of incorrectly shaped batches.
    """
    X = np.random.rand(
        30,
        63,
    ).astype(np.float32)

    pipeline = PreprocessingPipeline()

    with pytest.raises(ValueError):
        pipeline.process_batch(X)