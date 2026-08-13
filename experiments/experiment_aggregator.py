import csv
from pathlib import Path
from typing import Any, Dict, List

from experiments.metrics_loader import ExperimentMetricsLoader
from src.evaluation.model_comparison import ModelComparison


class ExperimentAggregator:
    """
    Aggregate, rank, and export HLI-01 experiment results.
    """

    @staticmethod
    def aggregate(
        experiments_dir,
        metric: str = "accuracy",
    ) -> List[Dict[str, Any]]:
        experiments_dir = Path(experiments_dir)

        records = ExperimentMetricsLoader.load(
            experiments_dir,
            metric=metric,
        )

        if not records:
            raise ValueError(
                f"No experiment metrics found for metric '{metric}'."
            )

        return ModelComparison.compare(
            records,
            metric=metric,
        )

    @staticmethod
    def export_csv(
        experiments_dir,
        output_path,
        metric: str = "accuracy",
    ) -> Path:
        """
        Export aggregated experiment results to a CSV file.
        """

        output_path = Path(output_path)

        records = ExperimentAggregator.aggregate(
            experiments_dir=experiments_dir,
            metric=metric,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = []

        for record in records:
            for key in record.keys():
                if key not in fieldnames:
                    fieldnames.append(key)

        with open(
            output_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )

            writer.writeheader()
            writer.writerows(records)

        return output_path
