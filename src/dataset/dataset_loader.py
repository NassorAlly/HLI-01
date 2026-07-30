"""
dataset_loader.py

Loads all gesture samples into memory.
"""

from pathlib import Path
import numpy as np

from .dataset_collector import DatasetCollector
from .label_manager import LabelManager


class DatasetLoader:
    """
    Loads all gesture samples into memory.
    """

    def __init__(self, dataset_path="dataset"):
        self.dataset_path = Path(dataset_path)

        self.collector = DatasetCollector(dataset_path)
        self.label_manager = LabelManager(
           self.collector.get_classes()
)

    def load(self):
        """
        Load the dataset.

        Returns
        -------
        X : numpy.ndarray
            Shape (N, 30, 63)

        y : numpy.ndarray
            Shape (N,)
        """

        X = []
        y = []

        samples = self.collector.collect()

        for sample in samples:

            class_name = sample["label"]
            sample_path = sample["path"]

            data = np.load(sample_path).astype(np.float32)

            label = self.label_manager.encode(class_name)

            X.append(data)
            y.append(label)

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)

        return X, y
