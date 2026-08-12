import json

from experiments.experiment_manager import ExperimentManager


def test_create_experiment(tmp_path):
    manager = ExperimentManager(base_dir=tmp_path)

    experiment_dir = manager.create_experiment("baseline")

    assert experiment_dir.exists()
    assert experiment_dir.is_dir()

    assert experiment_dir.name.startswith("EXP_")
    assert experiment_dir.name.endswith("_baseline")

    assert (experiment_dir / "figures").exists()
    assert (experiment_dir / "predictions").exists()
    assert (experiment_dir / "checkpoints").exists()

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------
    metadata_path = experiment_dir / "metadata.json"

    assert metadata_path.exists()
    assert metadata_path.is_file()

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["experiment_id"] == experiment_dir.name
    assert metadata["experiment_name"] == "baseline"
    assert metadata["project_version"] == "1.0.0"

    assert "created_at" in metadata
    assert "python_version" in metadata
    assert "platform" in metadata

    # --------------------------------------------------
    # Configuration snapshot
    # --------------------------------------------------
    config_path = experiment_dir / "config.json"

    assert config_path.exists()
    assert config_path.is_file()

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    assert "default" in config
    assert "dataset" in config
    assert "model" in config
    assert "training" in config

    assert config["model"]["model"]["name"] == "BiLSTM_Attention"
    assert config["dataset"]["dataset"]["sequence_length"] == 30
    assert config["training"]["training"]["epochs"] == 50