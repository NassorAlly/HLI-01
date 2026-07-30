import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.lstm_model import LSTMModel

print("Testing LSTM Model...")

model = LSTMModel(
    input_size=63,
    hidden_size=128,
    num_layers=2,
    num_classes=4
)

print("✓ Model created successfully")

dummy_input = torch.randn(8, 30, 63)

output = model(dummy_input)

print("✓ Forward pass successful")
print("Output Shape:", output.shape)

assert output.shape == (8, 4)

print("✓ Output shape is correct")
print("\nTEST PASSED")
