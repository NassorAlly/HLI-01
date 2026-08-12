from experiments.metadata import ExperimentMetadata


def test_generate_metadata():
    metadata = ExperimentMetadata.generate(
        experiment_id="EXP_TEST_001",
        experiment_name="baseline",
    )

    assert isinstance(metadata, dict)

    assert metadata["experiment_id"] == "EXP_TEST_001"
    assert metadata["experiment_name"] == "baseline"
    assert metadata["project_version"] == "1.0.0"

    assert "created_at" in metadata
    assert "python_version" in metadata
    assert "platform" in metadata


def test_custom_project_version():
    metadata = ExperimentMetadata.generate(
        experiment_id="EXP_TEST_002",
        experiment_name="ablation",
        project_version="1.0.1",
    )

    assert metadata["project_version"] == "1.0.1"