import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.models.lstm_model import LSTMModel
from src.training.checkpoint_manager import CheckpointManager

print("=" * 50)
print("Testing Checkpoint Manager")
print("=" * 50)

model = LSTMModel(
    input_size=63,
    hidden_size=128,
    num_layers=2,
    num_classes=4,
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

manager = CheckpointManager()

manager.save(
    model=model,
    optimizer=optimizer,
    epoch=5,
    loss=0.1234,
)

epoch, loss = manager.load(
    model=model,
    optimizer=optimizer,
)

print("Epoch :", epoch)
print("Loss  :", loss)

assert epoch == 5
assert abs(loss - 0.1234) < 1e-6

print("\n✓ Checkpoint Manager works correctly.")
print("TEST PASSED")
