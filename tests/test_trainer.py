import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.lstm_model import LSTMModel
from src.training.trainer import Trainer

print("=" * 50)
print("Testing Trainer")
print("=" * 50)

X = torch.randn(64, 30, 63)
y = torch.randint(0, 4, (64,))

dataset = TensorDataset(X, y)

train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

valid_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
)

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

device = torch.device("cpu")

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    device=device,
    epochs=2,
)

trainer.train()

print("\n✓ Trainer works correctly.")
print("TEST PASSED")
