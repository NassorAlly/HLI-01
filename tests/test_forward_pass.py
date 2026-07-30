import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.model_factory import ModelFactory

print("=" * 50)
print("Testing Complete Forward Pass")
print("=" * 50)

model = ModelFactory.create(
    model_name="lstm",
    input_size=63,
    hidden_size=128,
    num_layers=2,
    num_classes=4
)

batch = torch.randn(32, 30, 63)

prediction = model(batch)

print("Input Shape      :", batch.shape)
print("Output Shape     :", prediction.shape)

assert prediction.shape == (32, 4)

print("\n✓ Forward pass successful")
print("✓ Batch processing successful")
print("✓ Output dimensions correct")
print("\nALL MODEL TESTS PASSED")
