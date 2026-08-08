import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn


print("=" * 60)
print("HLI-01 v0.8.0 - Testing Learning Rate Scheduler")
print("=" * 60)


# --------------------------------------------------
# Simple model
# --------------------------------------------------

model = nn.Linear(
    10,
    4,
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

initial_lr = 0.001

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=initial_lr,
)


# --------------------------------------------------
# Scheduler
# --------------------------------------------------

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )
)


# --------------------------------------------------
# Simulated validation losses
# --------------------------------------------------

validation_losses = [
    1.00,
    1.00,
    1.00,
    1.00,
]


print()
print(
    "Initial learning rate:",
    f"{initial_lr:.6f}",
)

print()


# --------------------------------------------------
# Run scheduler
# --------------------------------------------------

learning_rates = []

for epoch, loss in enumerate(
    validation_losses,
    start=1,
):

    scheduler.step(loss)

    current_lr = (
        optimizer.param_groups[0]["lr"]
    )

    learning_rates.append(
        current_lr
    )

    print(
        f"Epoch {epoch} | "
        f"Validation Loss={loss:.4f} | "
        f"Learning Rate={current_lr:.6f}"
    )


# --------------------------------------------------
# Validate scheduler behavior
# --------------------------------------------------

final_lr = (
    optimizer.param_groups[0]["lr"]
)

assert final_lr < initial_lr

assert min(learning_rates) < initial_lr


# --------------------------------------------------
# Results
# --------------------------------------------------

print()
print("=" * 60)
print("Learning Rate Scheduler Test Results")
print("=" * 60)

print(
    "Initial learning rate:",
    f"{initial_lr:.6f}",
)

print(
    "Final learning rate  :",
    f"{final_lr:.6f}",
)

print()

print(
    "✓ Scheduler detected validation plateau."
)

print(
    "✓ Learning rate was reduced."
)

print(
    "✓ Optimizer received the new learning rate."
)

print()
print("TEST PASSED")