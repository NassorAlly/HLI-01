import json

import pytest

from experiments.metrics_loader import ExperimentMetricsLoader


def test_load_valid_experiment_metrics(tmp_path):
    experiment_dir = tmp_path / "EXP_VALID"
    experiment_dir.mkdir()

    metrics = {
        "test_accuracy": 0.95,
        "test_f1_score": 0.94,
    }

    with open(
        experiment_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file)

    records = ExperimentMetricsLoader.load(
        tmp_path,
        metric="test_accuracy",
    )

    assert len(records) == 1
    assert records[0]["name"] == "EXP_VALID"
    assert records[0]["test_accuracy"] == 0.95


def test_skip_experiment_without_metrics_file(tmp_path):
    experiment_dir = tmp_path / "EXP_NO_METRICS"
    experiment_dir.mkdir()

    records = ExperimentMetricsLoader.load(
        tmp_path,
        metric="test_accuracy",
    )

    assert records == []


def test_skip_experiment_missing_requested_metric(tmp_path):
    experiment_dir = tmp_path / "EXP_NO_TEST_METRIC"
    experiment_dir.mkdir()

    with open(
        experiment_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "best_validation_loss": 0.1,
            },
            file,
        )

    records = ExperimentMetricsLoader.load(
        tmp_path,
        metric="test_accuracy",
    )

    assert records == []


def test_missing_experiments_directory_raises_error(tmp_path):
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        ExperimentMetricsLoader.load(
            missing_dir,
            metric="test_accuracy",
        )
