"""
test_splitter.py

Unit tests for the HLI-01 DatasetSplitter.
"""

import numpy as np

from src.dataset.dataset_splitter import DatasetSplitter


class DummyLoader:
    """
    Minimal stand-in for DatasetLoader.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def load(self):
        return self.X, self.y


def create_balanced_dataset():
    """
    Create a balanced 4-class synthetic dataset.

    100 samples per class
    Total = 400 samples
    """

    X = np.random.rand(
        400,
        30,
        63,
    ).astype(np.float32)

    y = np.repeat(
        [0, 1, 2, 3],
        100,
    ).astype(np.int64)

    return X, y


def test_splitter_output_sizes():
    """
    Test expected train, validation and test sizes.
    """

    X, y = create_balanced_dataset()

    splitter = DatasetSplitter()
    splitter.loader = DummyLoader(X, y)

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = splitter.split()

    assert len(X_train) == 280
    assert len(X_valid) == 60
    assert len(X_test) == 60

    assert len(y_train) == 280
    assert len(y_valid) == 60
    assert len(y_test) == 60


def test_splitter_preserves_sample_shape():
    """
    Test that sequence dimensions are preserved.
    """

    X, y = create_balanced_dataset()

    splitter = DatasetSplitter()
    splitter.loader = DummyLoader(X, y)

    (
        X_train,
        X_valid,
        X_test,
        _,
        _,
        _,
    ) = splitter.split()

    assert X_train.shape[1:] == (30, 63)
    assert X_valid.shape[1:] == (30, 63)
    assert X_test.shape[1:] == (30, 63)


def test_splitter_preserves_total_samples():
    """
    Test that no samples are lost during splitting.
    """

    X, y = create_balanced_dataset()

    splitter = DatasetSplitter()
    splitter.loader = DummyLoader(X, y)

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = splitter.split()

    total_X = (
        len(X_train)
        + len(X_valid)
        + len(X_test)
    )

    total_y = (
        len(y_train)
        + len(y_valid)
        + len(y_test)
    )

    assert total_X == 400
    assert total_y == 400


def test_splitter_is_stratified():
    """
    Test that class distribution is preserved
    across train, validation and test sets.
    """

    X, y = create_balanced_dataset()

    splitter = DatasetSplitter()
    splitter.loader = DummyLoader(X, y)

    (
        _,
        _,
        _,
        y_train,
        y_valid,
        y_test,
    ) = splitter.split()

    train_counts = np.bincount(y_train)
    valid_counts = np.bincount(y_valid)
    test_counts = np.bincount(y_test)

    assert list(train_counts) == [70, 70, 70, 70]
    assert list(valid_counts) == [15, 15, 15, 15]
    assert list(test_counts) == [15, 15, 15, 15]


def test_splitter_is_reproducible():
    """
    Test that the fixed random seed produces
    identical splits across repeated runs.
    """

    X, y = create_balanced_dataset()

    splitter_1 = DatasetSplitter()
    splitter_1.loader = DummyLoader(X, y)

    result_1 = splitter_1.split()

    splitter_2 = DatasetSplitter()
    splitter_2.loader = DummyLoader(X, y)

    result_2 = splitter_2.split()

    for array_1, array_2 in zip(result_1, result_2):
        np.testing.assert_array_equal(
            array_1,
            array_2,
        )


def test_splitter_preserves_dtypes():
    """
    Test that data and label dtypes are preserved.
    """

    X, y = create_balanced_dataset()

    splitter = DatasetSplitter()
    splitter.loader = DummyLoader(X, y)

    (
        X_train,
        X_valid,
        X_test,
        y_train,
        y_valid,
        y_test,
    ) = splitter.split()

    assert X_train.dtype == np.float32
    assert X_valid.dtype == np.float32
    assert X_test.dtype == np.float32

    assert y_train.dtype == np.int64
    assert y_valid.dtype == np.int64
    assert y_test.dtype == np.int64