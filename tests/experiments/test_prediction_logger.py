import csv

from experiments.prediction_logger import PredictionLogger


def test_save_predictions(tmp_path):
    logger = PredictionLogger(tmp_path)

    predictions = [
        {
            "sample_id": 1,
            "true_label": "peace",
            "predicted_label": "peace",
            "confidence": 0.9990,
        },
        {
            "sample_id": 2,
            "true_label": "hello",
            "predicted_label": "hello",
            "confidence": 0.9820,
        },
    ]

    output_path = logger.save_predictions(predictions)

    assert output_path.exists()
    assert output_path.name == "predictions.csv"

    with open(output_path, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 2

    assert rows[0]["true_label"] == "peace"
    assert rows[0]["predicted_label"] == "peace"
    assert float(rows[0]["confidence"]) == 0.9990


def test_empty_predictions_raise_error(tmp_path):
    logger = PredictionLogger(tmp_path)

    try:
        logger.save_predictions([])
        assert False, "Expected ValueError"
    except ValueError:
        assert True