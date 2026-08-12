from pathlib import Path

import yaml


CONFIG_DIR = Path("configs")

EXPECTED_CONFIGS = [
    "default.yaml",
    "dataset.yaml",
    "model.yaml",
    "training.yaml",
]


def test_config_files_exist():
    for config_name in EXPECTED_CONFIGS:
        config_path = CONFIG_DIR / config_name
        assert config_path.exists(), f"Missing config file: {config_path}"


def test_config_files_load():
    for config_name in EXPECTED_CONFIGS:
        config_path = CONFIG_DIR / config_name

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        assert config is not None
        assert isinstance(config, dict)