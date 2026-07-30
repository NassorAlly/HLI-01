import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base_model import BaseModel
from src.models.base_model import BaseModel

print("Testing BaseModel...")

try:
    BaseModel()
except TypeError as e:
    print("PASS")
    print(e)
