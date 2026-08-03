"""
test_statistics.py

Unit tests for the HLI-01 DatasetStatistics component.
"""

import numpy as np

from src.dataset.dataset_statistics import DatasetStatistics


class DummyLoader:
    """
    Minimal stand-in for DatasetLoader.
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def load(self):
        return self.X, self.y


class DummyCollector:
    """
    Minimal stand-in for DatasetCollector.
    """

    def __init__(self, samples, classes):
        self._samples = samples
        self._classes = classes

    def collect(self):
        return self._samples

    def get_num_classes(self):
        return len(self._classes)


def create_dataset():
    """
    Create a balanced synthetic HLI-01 dataset.
    """

    X = np.zeros(
        (8, 30, 63),
        dtype=np.float32,
    )

    for index in range(8):
        X[index] = index

    y = np.array(
        [0, 1, 2, 3, 0, 1, 2, 3],
        dtype=np.int64,
    )

    samples = [
        {"label": "hello", "path": "hello_1.npy"},
        {"label": "no", "path": "no_1.npy"},
        {"label": "peace", "path": "peace_1.npy"},
        {"label": "yes", "path": "yes_1.npy"},
        {"label": "hello", "path": "hello_2.npy"},
        {"label": "no", "path": "no_2.npy"},
        {"label": "peace", "path": "peace_2.npy"},
        {"label": "yes", "path": "yes_2.npy"},
    ]

    classes = [
        "hello",
        "no",
        "peace",
        "yes",
    ]

    return X, y, samples, classes


def test_statistics_reports_number_of_classes(capsys):
    """
    Test that the number of classes is reported correctly.
    """

    X, y, samples, classes = create_dataset()

    statistics = DatasetStatistics()

    statistics.loader = DummyLoader(X, y)
    statistics.collector = DummyCollector(samples, classes)

    statistics.summary()

    output = capsys.readouterr().out

    assert "Number of Classes : 4" in output


def test_statistics_reports_total_samples(capsys):
    """
    Test that total sample count is reported correctly.
    """

    X, y, samples, classes = create_dataset()

    statistics = DatasetStatistics()

    statistics.loader = DummyLoader(X, y)
    statistics.collector = DummyCollector(samples, classes)

    statistics.summary()

    output = capsys.readouterr().out

    assert "Total Samples     : 8" in output


def test_statistics_reports_sequence_dimensions(capsys):
    """
    Test sequence length and feature count reporting.
    """

    X, y, samples, classes = create_dataset()

    statistics = DatasetStatistics()

    statistics.loader = DummyLoader(X, y)
    statistics.collector = DummyCollector(samples, classes)

    statistics.summary()

    output = capsys.readouterr().out

    assert "Sequence Length   : 30" in output
    assert "Features          : 63" in output


def test_statistics_reports_dtype(capsys):
    """
    Test reporting of dataset dtype.
    """

    X, y, samples, classes = create_dataset()

    statistics = DatasetStatistics()

    statistics.loader = DummyLoader(X, y)
    statistics.collector = DummyCollector(samples, classes)

    statistics.summary()

    output = capsys.readouterr().out

    assert "Data Type         : float32" in output


def test_statistics_reports_class_counts(capsys):
    """
    Test samples-per-class reporting.
    """

    X, y, samples, classes = create_dataset()

    statistics = DatasetStatistics()

    statistics.loader = DummyLoader(X, y)
    statistics.collector = DummyCollector(samples, classes)

    statistics.summary()

    output = capsys.readouterr().out

    assert "hello      : 2" in output
    assert "no         : 2" in output
    assert "peace      : 2" in output
    assert "yes        : 2" in output


def test_statistics_reports_numeric_statistics(capsys):
    """
    Test reporting of basic descriptive statistics.
    """

    X, y, samples, classes = create_dataset()

    statistics = DatasetStatistics()

    statistics.loader = DummyLoader(X, y)
    statistics.collector = DummyCollector(samples, classes)

    statistics.summary()

    output = capsys.readouterr().out

    assert "Minimum Value     : 0.000000" in output
    assert "Maximum Value     : 7.000000" in output
    assert "Mean              : 3.500000" in output


def test_statistics_header_is_printed(capsys):
    """
    Test that the HLI-01 statistics header is present.
    """

    X, y, samples, classes = create_dataset()

    statistics = DatasetStatistics()

    statistics.loader = DummyLoader(X, y)
    statistics.collector = DummyCollector(samples, classes)

    statistics.summary()

    output = capsys.readouterr().out

    assert "HLI-01 DATASET STATISTICS" in output
    assert "Samples per Class" in output
    assert "Data Statistics" in output