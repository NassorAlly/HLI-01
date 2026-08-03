"""
test_validator.py

Unit tests for the HLI-01 DatasetValidator.
"""

import numpy as np

from src.dataset.dataset_validator import DatasetValidator
from src.config.settings import SEQUENCE_LENGTH, NUM_FEATURES


class DummyCollector:
    """
    Minimal stand-in for DatasetCollector.
    """

    def __init__(self, samples):
        self._samples = samples

    def collect(self):
        return self._samples


def test_validator_accepts_valid_sample(tmp_path):
    """
    Test that a valid sample produces no validation errors.
    """

    sample = np.random.rand(
        SEQUENCE_LENGTH,
        NUM_FEATURES,
    ).astype(np.float32)

    sample_path = tmp_path / "valid_sample.npy"
    np.save(sample_path, sample)

    validator = DatasetValidator(dataset_path=tmp_path)

    validator.collector = DummyCollector(
        [
            {
                "label": "hello",
                "path": sample_path,
            }
        ]
    )

    errors = validator.validate()

    assert errors == []


def test_validator_detects_invalid_shape(tmp_path):
    """
    Test detection of an invalid sample shape.
    """

    sample = np.random.rand(
        SEQUENCE_LENGTH - 1,
        NUM_FEATURES,
    ).astype(np.float32)

    sample_path = tmp_path / "invalid_shape.npy"
    np.save(sample_path, sample)

    validator = DatasetValidator(dataset_path=tmp_path)

    validator.collector = DummyCollector(
        [
            {
                "label": "no",
                "path": sample_path,
            }
        ]
    )

    errors = validator.validate()

    assert len(errors) == 1
    assert "Invalid shape" in errors[0]


def test_validator_detects_nan_values(tmp_path):
    """
    Test detection of NaN values.
    """

    sample = np.random.rand(
        SEQUENCE_LENGTH,
        NUM_FEATURES,
    ).astype(np.float32)

    sample[0, 0] = np.nan

    sample_path = tmp_path / "nan_sample.npy"
    np.save(sample_path, sample)

    validator = DatasetValidator(dataset_path=tmp_path)

    validator.collector = DummyCollector(
        [
            {
                "label": "peace",
                "path": sample_path,
            }
        ]
    )

    errors = validator.validate()

    assert len(errors) == 1
    assert "Contains NaN values" in errors[0]


def test_validator_detects_infinite_values(tmp_path):
    """
    Test detection of Infinite values.
    """

    sample = np.random.rand(
        SEQUENCE_LENGTH,
        NUM_FEATURES,
    ).astype(np.float32)

    sample[0, 0] = np.inf

    sample_path = tmp_path / "inf_sample.npy"
    np.save(sample_path, sample)

    validator = DatasetValidator(dataset_path=tmp_path)

    validator.collector = DummyCollector(
        [
            {
                "label": "yes",
                "path": sample_path,
            }
        ]
    )

    errors = validator.validate()

    assert len(errors) == 1
    assert "Contains Infinite values" in errors[0]


def test_validator_detects_multiple_errors(tmp_path):
    """
    Test detection of multiple problems in the same sample.
    """

    sample = np.random.rand(
        SEQUENCE_LENGTH - 1,
        NUM_FEATURES,
    ).astype(np.float32)

    sample[0, 0] = np.nan
    sample[0, 1] = np.inf

    sample_path = tmp_path / "multiple_errors.npy"
    np.save(sample_path, sample)

    validator = DatasetValidator(dataset_path=tmp_path)

    validator.collector = DummyCollector(
        [
            {
                "label": "hello",
                "path": sample_path,
            }
        ]
    )

    errors = validator.validate()

    assert len(errors) == 3

    assert any("Invalid shape" in error for error in errors)
    assert any("Contains NaN values" in error for error in errors)
    assert any("Contains Infinite values" in error for error in errors)


def test_validator_handles_corrupt_file(tmp_path):
    """
    Test that unreadable or corrupt files are reported.
    """

    corrupt_path = tmp_path / "corrupt.npy"

    corrupt_path.write_text(
        "this is not a valid numpy file",
        encoding="utf-8",
    )

    validator = DatasetValidator(dataset_path=tmp_path)

    validator.collector = DummyCollector(
        [
            {
                "label": "no",
                "path": corrupt_path,
            }
        ]
    )

    errors = validator.validate()

    assert len(errors) == 1
    assert "no:" in errors[0]


def test_validator_handles_empty_dataset(tmp_path):
    """
    Test validation when the dataset contains no samples.
    """

    validator = DatasetValidator(dataset_path=tmp_path)

    validator.collector = DummyCollector([])

    errors = validator.validate()

    assert errors == []