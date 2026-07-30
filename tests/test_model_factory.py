import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import ModelFactory

print("Testing Model Factory...")

model = ModelFactory.create(
    model_name="lstm",
    input_size=63,
    hidden_size=128,
    num_layers=2,
    num_classes=4
)

print("✓ Model created successfully")
print("Model Name:", model.get_model_name())

assert model.get_model_name() == "BiLSTM_Attention"

print("\nTEST PASSED")
