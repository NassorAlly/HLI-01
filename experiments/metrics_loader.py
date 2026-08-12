import json
from pathlib import Path
from typing import Any, Dict, List


class ExperimentMetricsLoader:
    """
    Load saved HLI-01 experiment metrics for comparison.
    """

    @staticmethod
    def load(
        experiments_dir,
        metric: str,
    ) -> List[Dict[str, Any]]:
        """
        Load experiment metrics containing the requested metric.

        Parameters
        ----------
        experiments_dir : str or Path
            Directory containing experiment folders.

        metric : str
            Metric required for comparison.

        Returns
        -------
        list of dict
            Experiment records containing the requested metric.
        """

        experiments_dir = Path(experiments_dir)

        if not experiments_dir.exists():
            raise FileNotFoundError(
                f"Experiments directory not found: "
                f"{experiments_dir}"
            )

        records = []

        for experiment_dir in sorted(
            experiments_dir.iterdir()
        ):
            if not experiment_dir.is_dir():
                continue

            metrics_path = (
                experiment_dir / "metrics.json"
            )

            if not metrics_path.exists():
                continue

            with open(
                metrics_path,
                "r",
                encoding="utf-8",
            ) as file:
                metrics = json.load(file)

            if metric not in metrics:
                continue

            record = {
                "name": experiment_dir.name,
                "experiment_dir": str(
                    experiment_dir
                ),
            }

            record.update(metrics)

            records.append(record)

        return records
