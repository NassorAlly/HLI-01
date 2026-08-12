from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigLoader:
    """
    Load YAML configuration files for HLI-01.
    """

    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)

    def load(self, filename: str) -> Dict[str, Any]:
        config_path = self.config_dir / filename

        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}"
            )

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if config is None:
            return {}

        if not isinstance(config, dict):
            raise ValueError(
                f"Configuration file must contain a YAML mapping: {config_path}"
            )

        return config

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        return {
            "default": self.load("default.yaml"),
            "dataset": self.load("dataset.yaml"),
            "model": self.load("model.yaml"),
            "training": self.load("training.yaml"),
        }