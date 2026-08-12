import csv
from pathlib import Path
from typing import List, Dict, Any


class PredictionLogger:
    """
    Save HLI-01 model predictions for research analysis.
    """

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = Path(experiment_dir)

        if not self.experiment_dir.exists():
            raise FileNotFoundError(
                f"Experiment directory not found: {self.experiment_dir}"
            )

        self.predictions_dir = self.experiment_dir / "predictions"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)

    def save_predictions(
        self,
        predictions: List[Dict[str, Any]],
        filename: str = "predictions.csv",
    ) -> Path:
        """
        Save prediction records to CSV.
        """

        if not predictions:
            raise ValueError("Predictions cannot be empty.")

        output_path = self.predictions_dir / filename

        fieldnames = list(predictions[0].keys())

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerows(predictions)

        return output_path