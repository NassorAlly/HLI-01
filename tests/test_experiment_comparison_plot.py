from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from src.visualization.experiment_comparison_plot import (
    ExperimentComparisonPlotter,
)


def test_experiment_comparison_plot(tmp_path):
    experiments = [
        {
            "name": "LSTM",
            "accuracy": 0.92,
            "f1_macro": 0.90,
        },
        {
            "name": "BiLSTM",
            "accuracy": 0.95,
            "f1_macro": 0.93,
        },
        {
            "name": "BiLSTM_Attention",
            "accuracy": 0.98,
            "f1_macro": 0.97,
        },
    ]

    plotter = ExperimentComparisonPlotter(
        save_dir=tmp_path,
    )

    output_path = plotter.plot(
        experiments=experiments,
        metrics=["accuracy", "f1_macro"],
        filename="experiment_comparison.png",
    )

    output_path = Path(output_path)

    assert output_path.exists()
    assert output_path.name == "experiment_comparison.png"
    assert output_path.stat().st_size > 0


def test_experiment_comparison_plot_rejects_empty_experiments(tmp_path):
    plotter = ExperimentComparisonPlotter(
        save_dir=tmp_path,
    )

    try:
        plotter.plot(
            experiments=[],
            metrics=["accuracy"],
        )
    except ValueError as error:
        assert "Experiments cannot be empty" in str(error)
    else:
        raise AssertionError(
            "Expected ValueError for empty experiments."
        )


def test_experiment_comparison_plot_rejects_missing_metric(tmp_path):
    experiments = [
        {
            "name": "LSTM",
            "accuracy": 0.92,
        }
    ]

    plotter = ExperimentComparisonPlotter(
        save_dir=tmp_path,
    )

    try:
        plotter.plot(
            experiments=experiments,
            metrics=["accuracy", "f1_macro"],
        )
    except KeyError as error:
        assert "f1_macro" in str(error)
    else:
        raise AssertionError(
            "Expected KeyError for missing metric."
        )
