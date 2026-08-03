"""
test_loader.py

Unit tests for the HLI-01 DatasetLoader.
"""

import numpy as np

from src.dataset.dataset_loader import DatasetLoader


class DummyCollector:
    """
    Minimal stand-in for DatasetCollector used during testing.
    """

    def __init__(self, samples, classes):
        self._samples = samples
        self._classes = classes

    def collect(self):
        return self._samples

    def get_classes(self):
        return self._classes


def test_loader_loads_dataset(tmp_path):
    """
    Test that DatasetLoader correctly loads gesture samples
    and returns arrays with the expected shapes and dtypes.
    """

    sample_hello = np.random.rand(30, 63).astype(np.float32)
    sample_no = np.random.rand(30, 63).astype(np.float32)

    hello_path = tmp_path / "hello_001.npy"
    no_path = tmp_path / "no_001.npy"

    np.save(hello_path, sample_hello)
    np.save(no_path, sample_no)

    loader = DatasetLoader(dataset_path=tmp_path)

    loader.collector = DummyCollector(
        samples=[
            {"label": "hello", "path": hello_path},
            {"label": "no", "path": no_path},
        ],
        classes=["hello", "no"],
    )

    from src.dataset.label_manager import LabelManager

    loader.label_manager = LabelManager(["hello", "no"])

    X, y = loader.load()

    assert X.shape == (2, 30, 63)
    assert y.shape == (2,)

    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_loader_preserves_sample_values(tmp_path):
    """
    Test that loaded sample values match the saved .npy files.
    """

    sample = np.random.rand(30, 63).astype(np.float32)

    sample_path = tmp_path / "peace_001.npy"
    np.save(sample_path, sample)

    loader = DatasetLoader(dataset_path=tmp_path)

    loader.collector = DummyCollector(
        samples=[
            {"label": "peace", "path": sample_path},
        ],
        classes=["peace"],
    )

    from src.dataset.label_manager import LabelManager

    loader.label_manager = LabelManager(["peace"])

    X, y = loader.load()

    np.testing.assert_allclose(
        X[0],
        sample,
        rtol=1e-6,
        atol=1e-6,
    )

    assert y[0] == 0


def test_loader_encodes_multiple_classes(tmp_path):
    """
    Test that class names are encoded into integer labels.
    """

    classes = ["hello", "no", "peace", "yes"]

    samples = []

    for index, class_name in enumerate(classes):
        data = np.full(
            (30, 63),
            fill_value=index,
            dtype=np.float32,
        )

        sample_path = tmp_path / f"{class_name}_001.npy"
        np.save(sample_path, data)

        samples.append(
            {
                "label": class_name,
                "path": sample_path,
            }
        )

    loader = DatasetLoader(dataset_path=tmp_path)

    loader.collector = DummyCollector(
        samples=samples,
        classes=classes,
    )

    from src.dataset.label_manager import LabelManager

    loader.label_manager = LabelManager(classes)

    X, y = loader.load()

    assert X.shape == (4, 30, 63)
    assert y.shape == (4,)

    assert list(y) == [0, 1, 2, 3]


def test_loader_handles_empty_dataset(tmp_path):
    """
    Test loader behaviour when no samples are available.
    """

    loader = DatasetLoader(dataset_path=tmp_path)

    loader.collector = DummyCollector(
        samples=[],
        classes=[],
    )

    from src.dataset.label_manager import LabelManager

    loader.label_manager = LabelManager([])

    X, y = loader.load()

    assert X.shape == (0,)
    assert y.shape == (0,)

    assert X.dtype == np.float32
    assert y.dtype == np.int64