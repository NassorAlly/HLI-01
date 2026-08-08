import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.data_loader import DataLoaderManager


print("=" * 60)
print("HLI-01 v0.8.0 - Testing DataLoaderManager")
print("=" * 60)


# --------------------------------------------------
# Create DataLoader Manager
# --------------------------------------------------

manager = DataLoaderManager()


# --------------------------------------------------
# Create DataLoaders
# --------------------------------------------------

train_loader, valid_loader, test_loader = manager.create()


# --------------------------------------------------
# Display batch information
# --------------------------------------------------

print()

print("Training Batches  :", len(train_loader))
print("Validation Batches:", len(valid_loader))
print("Testing Batches   :", len(test_loader))


# --------------------------------------------------
# Inspect one training batch
# --------------------------------------------------

X, y = next(iter(train_loader))


print()

print("Batch Shape       :", X.shape)
print("Labels Shape      :", y.shape)
print("Data Type         :", X.dtype)


# --------------------------------------------------
# Basic validation
# --------------------------------------------------

assert len(train_loader) > 0
assert len(valid_loader) > 0
assert len(test_loader) > 0

assert X.ndim == 3
assert X.shape[1] == 30
assert X.shape[2] == 63

assert y.ndim == 1
assert X.shape[0] == y.shape[0]

assert str(X.dtype) == "torch.float32"


# --------------------------------------------------
# Test result
# --------------------------------------------------

print()

print("✓ Training DataLoader created correctly.")
print("✓ Validation DataLoader created correctly.")
print("✓ Testing DataLoader created correctly.")
print("✓ Batch dimensions are correct.")
print("✓ Feature dimensions are correct.")
print("✓ Data type is correct.")

print()
print("TEST PASSED")

print("=" * 60)