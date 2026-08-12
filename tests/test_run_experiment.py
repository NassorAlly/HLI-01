from pathlib import Path

import run_experiment


def test_initialize_experiment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_experiment.ExperimentManager,
        "__init__",
        lambda self: setattr(self, "base_dir", Path(tmp_path)),
    )

    def create_experiment(self, name=None):
        experiment_dir = self.base_dir / "EXP_TEST_baseline"

        experiment_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return experiment_dir

    monkeypatch.setattr(
        run_experiment.ExperimentManager,
        "create_experiment",
        create_experiment,
    )

    experiment_dir, logger = (
        run_experiment.initialize_experiment(
            experiment_name="baseline",
            seed=42,
        )
    )

    assert experiment_dir.exists()
    assert experiment_dir.name == "EXP_TEST_baseline"
    assert logger.experiment_dir == experiment_dir


def test_run_experiment_connects_training(
    tmp_path,
    monkeypatch,
):
    experiment_dir = (
        Path(tmp_path)
        / "EXP_TEST_training"
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    class DummyLogger:
        def __init__(self):
            self.saved_history = None
            self.saved_metrics = None

        def save_training_history(self, history):
            self.saved_history = history

        def save_metrics(self, metrics):
            self.saved_metrics = metrics

    dummy_logger = DummyLogger()

    monkeypatch.setattr(
        run_experiment,
        "initialize_experiment",
        lambda experiment_name, seed: (
            experiment_dir,
            dummy_logger,
        ),
    )

    training_call = {}

    def fake_run_training(
        checkpoint_dir,
        figure_dir,
        seed,
    ):
        training_call["checkpoint_dir"] = (
            Path(checkpoint_dir)
        )
        training_call["figure_dir"] = (
            Path(figure_dir)
        )
        training_call["seed"] = seed

        history = {
            "train_loss": [1.0, 0.5],
            "valid_loss": [1.1, 0.6],
        }

        summary = {
            "best_epoch": 2,
            "best_validation_loss": 0.6,
        }

        return history, summary

    monkeypatch.setattr(
        run_experiment,
        "run_training",
        fake_run_training,
    )

    (
        returned_dir,
        returned_logger,
        history,
        summary,
    ) = run_experiment.run_experiment(
        experiment_name="baseline",
        seed=42,
    )

    assert returned_dir == experiment_dir
    assert returned_logger is dummy_logger

    assert (
        training_call["checkpoint_dir"]
        == experiment_dir / "checkpoints"
    )

    assert (
        training_call["figure_dir"]
        == experiment_dir / "figures"
    )

    assert training_call["seed"] == 42

    assert history["train_loss"] == [1.0, 0.5]
    assert summary["best_epoch"] == 2

    assert dummy_logger.saved_history == history
    assert dummy_logger.saved_metrics == summary
