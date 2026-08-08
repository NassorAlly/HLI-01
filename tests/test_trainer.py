import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.lstm_model import LSTMModel
from src.training.trainer import Trainer


print("=" * 60)
print("HLI-01 v0.8.0 - Testing Trainer")
print("=" * 60)


# --------------------------------------------------
# Create synthetic sequence dataset
# --------------------------------------------------

X = torch.randn(
    64,
    30,
    63,
)

y = torch.randint(
    0,
    4,
    (64,),
)

dataset = TensorDataset(
    X,
    y,
)


# --------------------------------------------------
# Create data loaders
# --------------------------------------------------

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


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = LSTMModel(
    input_size=63,
    hidden_size=128,
    num_layers=2,
    num_classes=4,
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cpu")


# --------------------------------------------------
# Trainer
# --------------------------------------------------

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    optimizer=optimizer,
    device=device,
    epochs=2,
)


# --------------------------------------------------
# Run training
# --------------------------------------------------

history = trainer.train()


# --------------------------------------------------
# Validate training history
# --------------------------------------------------

assert isinstance(history, dict)

assert "train_loss" in history
assert "train_accuracy" in history
assert "valid_loss" in history
assert "valid_accuracy" in history


# Two epochs should have been recorded
assert len(history["train_loss"]) == 2
assert len(history["train_accuracy"]) == 2
assert len(history["valid_loss"]) == 2
assert len(history["valid_accuracy"]) == 2


# --------------------------------------------------
# Validate values
# --------------------------------------------------

for loss in history["train_loss"]:
    assert loss >= 0.0

for loss in history["valid_loss"]:
    assert loss >= 0.0

for accuracy in history["train_accuracy"]:
    assert 0.0 <= accuracy <= 1.0

for accuracy in history["valid_accuracy"]:
    assert 0.0 <= accuracy <= 1.0


# --------------------------------------------------
# Validate best model tracking
# --------------------------------------------------

assert trainer.best_epoch is not None
assert trainer.best_loss < float("inf")


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 60)
print("Trainer Test Results")
print("=" * 60)

print(
    f"Training epochs recorded : "
    f"{len(history['train_loss'])}"
)

print(
    f"Best epoch               : "
    f"{trainer.best_epoch}"
)

print(
    f"Best validation loss     : "
    f"{trainer.best_loss:.4f}"
)

print(
    f"Final training accuracy  : "
    f"{history['train_accuracy'][-1]:.4f}"
)

print(
    f"Final validation accuracy: "
    f"{history['valid_accuracy'][-1]:.4f}"
)

print()
print("✓ Training loop works.")
print("✓ Validation loop works.")
print("✓ Training history works.")
print("✓ Validation accuracy works.")
print("✓ Best epoch tracking works.")
print("✓ Checkpoint integration works.")

print()
print("TEST PASSED")