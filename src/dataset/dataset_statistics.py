"""
dataset_statistics.py

Computes descriptive statistics for the sign language dataset.
"""

import numpy as np

from .dataset_loader import DatasetLoader
from .dataset_collector import DatasetCollector


class DatasetStatistics:
    """
    Computes dataset statistics.
    """

    def __init__(self, dataset_path="dataset"):

        self.loader = DatasetLoader(dataset_path)
        self.collector = DatasetCollector(dataset_path)

    def summary(self):

        X, y = self.loader.load()

        print("=" * 60)
        print("HLI-01 DATASET STATISTICS")
        print("=" * 60)

        print(f"Number of Classes : {self.collector.get_num_classes()}")
        print(f"Total Samples     : {len(X)}")
        print(f"Sequence Length   : {X.shape[1]}")
        print(f"Features          : {X.shape[2]}")
        print(f"Data Type         : {X.dtype}")

        print("\nSamples per Class")
        print("-" * 30)

        samples = self.collector.collect()

        class_counts = {}

        for sample in samples:

            label = sample["label"]

            if label not in class_counts:
                class_counts[label] = 0

            class_counts[label] += 1

        for label in sorted(class_counts.keys()):
            print(f"{label:<10} : {class_counts[label]}")

        print("\nData Statistics")
        print("-" * 30)

        print(f"Minimum Value     : {X.min():.6f}")
        print(f"Maximum Value     : {X.max():.6f}")
        print(f"Mean              : {X.mean():.6f}")
        print(f"Standard Deviation: {X.std():.6f}")

        print("=" * 60)
