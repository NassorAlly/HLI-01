"""
test_dataset.py

Unit tests for the HLI-01 SignDataset.
"""

import numpy as np
import torch

from src.dataset.sign_dataset import SignDataset


def create_dummy_dataset():
    """
    Create a small synthetic dataset for testing.

    Shape:
        samples = 8
        sequence_length = 30
        features = 63
    """
    X = np.random.rand(8, 30, 63).astype(np.float32)

    y = np.array(
        [0, 1, 2, 3, 0, 1, 2, 3],
        dtype=np.int64,
    )

    return X, y


def test_dataset_length():
    """
    Test that dataset length matches number of samples.
    """
    X, y = create_dummy_dataset()

    dataset = SignDataset(X, y)

    assert len(dataset) == 8


def test_dataset_sample_shape():
    """
    Test that each sample has the expected HLI-01 shape.
    """
    X, y = create_dummy_dataset()

    dataset = SignDataset(X, y)

    sample, label = dataset[0]

    assert sample.shape == (30, 63)
    assert label.ndim == 0


def test_dataset_tensor_types():
    """
    Test tensor data types.
    """
    X, y = create_dummy_dataset()

    dataset = SignDataset(X, y)

    sample, label = dataset[0]

    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)

    assert sample.dtype == torch.float32
    assert label.dtype == torch.long


def test_dataset_input_shape():
    """
    Test the input_shape property.
    """
    X, y = create_dummy_dataset()

    dataset = SignDataset(X, y)

    assert dataset.input_shape == (30, 63)


def test_dataset_num_classes():
    """
    Test detection of the number of unique classes.
    """
    X, y = create_dummy_dataset()

    dataset = SignDataset(X, y)

    assert dataset.num_classes == 4


def test_dataset_values_preserved():
    """
    Test that conversion to tensors preserves sample values.
    """
    X, y = create_dummy_dataset()

    dataset = SignDataset(X, y)

    sample, label = dataset[3]

    np.testing.assert_allclose(
        sample.numpy(),
        X[3],
        rtol=1e-6,
        atol=1e-6,
    )

    assert label.item() == y[3]