import pytest

from src.evaluation.ablation import AblationStudy


def test_generate_attention_ablation_variants():
    base_config = {
        "model": {
            "name": "BiLSTM_Attention",
            "use_attention": True,
            "hidden_size": 128,
        }
    }

    variants = AblationStudy.generate_variants(
        base_config=base_config,
        parameter_path=["model", "use_attention"],
        values=[True, False],
    )

    assert len(variants) == 2

    assert variants[0]["model"]["use_attention"] is True
    assert variants[1]["model"]["use_attention"] is False

    assert base_config["model"]["use_attention"] is True


def test_empty_parameter_path_raises_error():
    with pytest.raises(ValueError):
        AblationStudy.generate_variants(
            base_config={"model": {}},
            parameter_path=[],
            values=[True, False],
        )


def test_empty_values_raise_error():
    with pytest.raises(ValueError):
        AblationStudy.generate_variants(
            base_config={"model": {"use_attention": True}},
            parameter_path=["model", "use_attention"],
            values=[],
        )


def test_missing_config_key_raises_error():
    with pytest.raises(KeyError):
        AblationStudy.generate_variants(
            base_config={"model": {}},
            parameter_path=["model", "use_attention"],
            values=[True, False],
        )