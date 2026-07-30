import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.early_stopping import EarlyStopping

print("=" * 50)
print("Testing EarlyStopping")
print("=" * 50)

early_stopping = EarlyStopping(
    patience=3,
    min_delta=0.01,
)

losses = [
    1.00,
    0.80,
    0.70,
    0.69,
    0.69,
    0.69,
    0.69,
]

for epoch, loss in enumerate(losses, start=1):

    stop = early_stopping(loss)

    print(
        f"Epoch {epoch} | "
        f"Loss={loss:.2f} | "
        f"Counter={early_stopping.counter} | "
        f"Stop={stop}"
    )

    if stop:
        break

assert stop is True

print("\n✓ EarlyStopping works correctly.")
print("TEST PASSED")
