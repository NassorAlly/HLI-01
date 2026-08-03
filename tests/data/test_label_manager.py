"""
test_label_manager.py

Unit tests for the HLI-01 LabelManager.
"""

import pytest

from src.dataset.label_manager import LabelManager


def test_label_manager_sorts_classes():
    """
    Test that class names are stored alphabetically.
    """

    manager = LabelManager(
        ["yes", "hello", "peace", "no"]
    )

    assert manager.get_labels() == [
        "hello",
        "no",
        "peace",
        "yes",
    ]


def test_label_manager_encodes_labels():
    """
    Test conversion of class names to numeric IDs.
    """

    manager = LabelManager(
        ["hello", "no", "peace", "yes"]
    )

    assert manager.encode("hello") == 0
    assert manager.encode("no") == 1
    assert manager.encode("peace") == 2
    assert manager.encode("yes") == 3


def test_label_manager_decodes_labels():
    """
    Test conversion of numeric IDs back to class names.
    """

    manager = LabelManager(
        ["hello", "no", "peace", "yes"]
    )

    assert manager.decode(0) == "hello"
    assert manager.decode(1) == "no"
    assert manager.decode(2) == "peace"
    assert manager.decode(3) == "yes"


def test_label_manager_encode_decode_round_trip():
    """
    Test that encoding followed by decoding
    returns the original class name.
    """

    labels = [
        "hello",
        "no",
        "peace",
        "yes",
    ]

    manager = LabelManager(labels)

    for label in labels:
        encoded = manager.encode(label)
        decoded = manager.decode(encoded)

        assert decoded == label


def test_label_manager_num_classes():
    """
    Test reporting of the number of classes.
    """

    manager = LabelManager(
        ["hello", "no", "peace", "yes"]
    )

    assert manager.get_num_classes() == 4


def test_label_manager_mapping_consistency():
    """
    Test internal forward and reverse mappings.
    """

    manager = LabelManager(
        ["yes", "peace", "hello", "no"]
    )

    assert manager.label_to_id == {
        "hello": 0,
        "no": 1,
        "peace": 2,
        "yes": 3,
    }

    assert manager.id_to_label == {
        0: "hello",
        1: "no",
        2: "peace",
        3: "yes",
    }


def test_label_manager_unknown_label_raises_error():
    """
    Test that encoding an unknown label raises KeyError.
    """

    manager = LabelManager(
        ["hello", "no", "peace", "yes"]
    )

    with pytest.raises(KeyError):
        manager.encode("unknown")


def test_label_manager_unknown_id_raises_error():
    """
    Test that decoding an unknown numeric ID raises KeyError.
    """

    manager = LabelManager(
        ["hello", "no", "peace", "yes"]
    )

    with pytest.raises(KeyError):
        manager.decode(99)