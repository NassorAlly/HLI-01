"""
early_stopping.py

Stops training when validation loss
does not improve.
"""

import numpy as np


class EarlyStopping:
    """
    Early stopping utility.
    """

    def __init__(
        self,
        patience=10,
        min_delta=0.0,
    ):

        self.patience = patience
        self.min_delta = min_delta

        self.best_loss = np.inf
        self.counter = 0

        self.stop = False

    def __call__(self, validation_loss):

        if validation_loss < self.best_loss - self.min_delta:

            self.best_loss = validation_loss
            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.stop = True

        return self.stop
