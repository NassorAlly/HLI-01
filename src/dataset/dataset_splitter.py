"""
dataset_splitter.py

Splits the dataset into training, validation and testing sets.
"""

from sklearn.model_selection import train_test_split

from .dataset_loader import DatasetLoader
from src.config.settings import (
    DATASET_PATH,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


class DatasetSplitter:
    """
    Splits the dataset into train, validation and test sets.
    """

    def __init__(self, dataset_path=DATASET_PATH):
        self.loader = DatasetLoader(dataset_path)

    def split(self):

        X, y = self.loader.load()

        # First split: Train / Temporary
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            train_size=TRAIN_RATIO,
            stratify=y,
            random_state=RANDOM_SEED,
        )

        # Second split: Validation / Test
        valid_ratio = VALID_RATIO / (VALID_RATIO + TEST_RATIO)

        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            train_size=valid_ratio,
            stratify=y_temp,
            random_state=RANDOM_SEED,
        )

        return (
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
        )
