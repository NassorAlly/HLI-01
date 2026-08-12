import random

import numpy as np
import torch

from src.utils.reproducibility import set_reproducibility


def test_reproducibility_same_seed():
    set_reproducibility(seed=42)

    python_value_1 = random.random()
    numpy_value_1 = np.random.rand()
    torch_value_1 = torch.rand(1).item()

    set_reproducibility(seed=42)

    python_value_2 = random.random()
    numpy_value_2 = np.random.rand()
    torch_value_2 = torch.rand(1).item()

    assert python_value_1 == python_value_2
    assert numpy_value_1 == numpy_value_2
    assert torch_value_1 == torch_value_2


def test_reproducibility_different_seed():
    set_reproducibility(seed=42)

    value_1 = torch.rand(1).item()

    set_reproducibility(seed=123)

    value_2 = torch.rand(1).item()

    assert value_1 != value_2