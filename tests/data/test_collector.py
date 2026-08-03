"""
test_collector.py

Unit tests for the HLI-01 DatasetCollector.
"""

from pathlib import Path

from src.dataset.dataset_collector import DatasetCollector


def create_sample_dataset(root):
    """
    Create a temporary HLI-01-style directory structure.
    """

    classes = [
        "hello",
        "no",
        "peace",
        "yes",
    ]

    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir()

        for index in range(2):
            sample_file = class_dir / f"{index}.npy"
            sample_file.write_bytes(b"dummy")

    return classes


def test_collector_gets_sorted_classes(tmp_path):
    """
    Test that class names are returned alphabetically.
    """

    create_sample_dataset(tmp_path)

    collector = DatasetCollector(tmp_path)

    assert collector.get_classes() == [
        "hello",
        "no",
        "peace",
        "yes",
    ]


def test_collector_reports_num_classes(tmp_path):
    """
    Test number of discovered classes.
    """

    create_sample_dataset(tmp_path)

    collector = DatasetCollector(tmp_path)

    assert collector.get_num_classes() == 4


def test_collector_collects_all_samples(tmp_path):
    """
    Test that all files are discovered.
    """

    create_sample_dataset(tmp_path)

    collector = DatasetCollector(tmp_path)

    samples = collector.collect()

    assert len(samples) == 8


def test_collector_sample_structure(tmp_path):
    """
    Test structure of collected sample records.
    """

    create_sample_dataset(tmp_path)

    collector = DatasetCollector(tmp_path)

    samples = collector.collect()

    for sample in samples:
        assert "label" in sample
        assert "path" in sample

        assert isinstance(sample["label"], str)
        assert isinstance(sample["path"], str)


def test_collector_assigns_correct_labels(tmp_path):
    """
    Test that collected files receive the correct class label.
    """

    create_sample_dataset(tmp_path)

    collector = DatasetCollector(tmp_path)

    samples = collector.collect()

    labels = [sample["label"] for sample in samples]

    assert labels.count("hello") == 2
    assert labels.count("no") == 2
    assert labels.count("peace") == 2
    assert labels.count("yes") == 2


def test_collector_paths_exist(tmp_path):
    """
    Test that returned paths point to real files.
    """

    create_sample_dataset(tmp_path)

    collector = DatasetCollector(tmp_path)

    samples = collector.collect()

    for sample in samples:
        assert Path(sample["path"]).exists()


def test_collector_reports_num_samples(tmp_path):
    """
    Test reporting of total sample count.
    """

    create_sample_dataset(tmp_path)

    collector = DatasetCollector(tmp_path)

    assert collector.get_num_samples() == 8


def test_collector_ignores_nested_directories_as_samples(tmp_path):
    """
    Test that directories inside class folders are not collected as files.
    """

    create_sample_dataset(tmp_path)

    nested = tmp_path / "hello" / "extra_folder"
    nested.mkdir()

    collector = DatasetCollector(tmp_path)

    samples = collector.collect()

    assert len(samples) == 8


def test_collector_handles_empty_class_folder(tmp_path):
    """
    Test behaviour when one class contains no sample files.
    """

    (tmp_path / "hello").mkdir()
    (tmp_path / "no").mkdir()

    sample_file = tmp_path / "no" / "0.npy"
    sample_file.write_bytes(b"dummy")

    collector = DatasetCollector(tmp_path)

    assert collector.get_num_classes() == 2
    assert collector.get_num_samples() == 1


def test_collector_handles_empty_dataset(tmp_path):
    """
    Test behaviour for an empty dataset directory.
    """

    collector = DatasetCollector(tmp_path)

    assert collector.get_classes() == []
    assert collector.get_num_classes() == 0
    assert collector.collect() == []
    assert collector.get_num_samples() == 0