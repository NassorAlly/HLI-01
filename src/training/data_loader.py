"""
data_loader.py

Creates PyTorch DataLoaders for training,
validation and testing.
"""

from torch.utils.data import DataLoader

from src.dataset.dataset_splitter import DatasetSplitter
from src.dataset.sign_dataset import SignDataset

from src.config.settings import BATCH_SIZE


class DataLoaderManager:
    """
    Creates PyTorch DataLoaders.
    """

    def __init__(self):

        splitter = DatasetSplitter()

        (
            X_train,
            X_valid,
            X_test,
            y_train,
            y_valid,
            y_test,
        ) = splitter.split()

        self.train_dataset = SignDataset(
            X_train,
            y_train,
        )

        self.valid_dataset = SignDataset(
            X_valid,
            y_valid,
        )

        self.test_dataset = SignDataset(
            X_test,
            y_test,
        )

    def create(self):
        """
        Create train, validation and test DataLoaders.
        """

        train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        valid_loader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        return train_loader, valid_loader, test_loader
