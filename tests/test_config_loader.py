from src.utils.config_loader import ConfigLoader


def test_load_single_config():
    loader = ConfigLoader()

    config = loader.load("model.yaml")

    assert isinstance(config, dict)
    assert "model" in config
    assert config["model"]["name"] == "BiLSTM_Attention"


def test_load_all_configs():
    loader = ConfigLoader()

    configs = loader.load_all()

    assert "default" in configs
    assert "dataset" in configs
    assert "model" in configs
    assert "training" in configs

    assert configs["dataset"]["dataset"]["sequence_length"] == 30
    assert configs["training"]["training"]["epochs"] == 50


def test_missing_config_raises_error():
    loader = ConfigLoader()

    try:
        loader.load("missing.yaml")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        assert True