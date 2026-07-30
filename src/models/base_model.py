"""
Base model class for HLI-01.
"""

from abc import ABC, abstractmethod
import torch.nn as nn


class BaseModel(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def get_model_name(self):
        pass
