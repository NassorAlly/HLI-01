"""
sign_dataset.py

Custom PyTorch Dataset for HLI-01.
"""

import torch
from torch.utils.data import Dataset


class SignDataset(Dataset):
    """
    PyTorch Dataset for sign language samples.
    """

    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray
            Gesture samples.

        y : numpy.ndarray
            Gesture labels.
        """

        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        """
        Return the number of samples.
        """
        return len(self.X)

    def __getitem__(self, index):
        """
        Return one sample and its label.
        """
        return self.X[index], self.y[index]

    @property
    def input_shape(self):
        """
        Return the shape of one sample.
        """
        return tuple(self.X.shape[1:])

    @property
    def num_classes(self):
        """
        Return the number of unique classes.
        """
        return len(torch.unique(self.y))
