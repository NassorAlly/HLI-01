import json

from experiments.experiment_logger import ExperimentLogger


def test_save_metrics(tmp_path):
    logger = ExperimentLogger(tmp_path)

    metrics = {
        "accuracy": 0.9833,
        "precision": 0.9800,
        "recall": 0.9833,
        "f1_score": 0.9815,
    }

    metrics_path = logger.save_metrics(metrics)

    assert metrics_path.exists()

    with open(metrics_path, "r", encoding="utf-8") as file:
        saved_metrics = json.load(file)

    assert saved_metrics == metrics


def test_save_training_history(tmp_path):
    logger = ExperimentLogger(tmp_path)

    history = {
        "train_loss": [0.8, 0.4, 0.2],
        "val_loss": [0.7, 0.3, 0.1],
        "train_accuracy": [0.70, 0.85, 0.95],
        "val_accuracy": [0.72, 0.88, 0.96],
    }

    history_path = logger.save_training_history(history)

    assert history_path.exists()

    with open(history_path, "r", encoding="utf-8") as file:
        saved_history = json.load(file)

    assert saved_history == history