import json

from experiments.experiment_aggregator import ExperimentAggregator


def test_aggregate_experiments(tmp_path):
    exp1 = tmp_path / "EXP_model_a"
    exp2 = tmp_path / "EXP_model_b"

    exp1.mkdir()
    exp2.mkdir()

    with open(exp1 / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": 0.90,
                "f1_macro": 0.88,
            },
            file,
        )

    with open(exp2 / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": 0.95,
                "f1_macro": 0.93,
            },
            file,
        )

    results = ExperimentAggregator.aggregate(
        tmp_path,
        metric="accuracy",
    )

    assert len(results) == 2
    assert results[0]["name"] == "EXP_model_b"
    assert results[0]["accuracy"] == 0.95
    assert results[1]["name"] == "EXP_model_a"


def test_export_aggregated_results_to_csv(tmp_path):
    experiments_dir = tmp_path / "experiments"
    experiments_dir.mkdir()

    exp1 = experiments_dir / "EXP_model_a"
    exp2 = experiments_dir / "EXP_model_b"

    exp1.mkdir()
    exp2.mkdir()

    with open(exp1 / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": 0.90,
                "f1_macro": 0.88,
            },
            file,
        )

    with open(exp2 / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "accuracy": 0.95,
                "f1_macro": 0.93,
            },
            file,
        )

    output_path = tmp_path / "comparison.csv"

    saved_path = ExperimentAggregator.export_csv(
        experiments_dir=experiments_dir,
        output_path=output_path,
        metric="accuracy",
    )

    assert saved_path.exists()

    content = saved_path.read_text(encoding="utf-8")

    assert "EXP_model_b" in content
    assert "EXP_model_a" in content
    assert "accuracy" in content
    assert "f1_macro" in content
