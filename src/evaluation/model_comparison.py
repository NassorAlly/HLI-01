from typing import Any, Dict, List


class ModelComparison:
    """
    Compare HLI-01 experimental results across models or runs.
    """

    @staticmethod
    def compare(
        experiments: List[Dict[str, Any]],
        metric: str = "accuracy",
    ) -> List[Dict[str, Any]]:
        if not experiments:
            raise ValueError("Experiments cannot be empty.")

        for experiment in experiments:
            if metric not in experiment:
                raise KeyError(
                    f"Metric '{metric}' missing from experiment."
                )

        return sorted(
            experiments,
            key=lambda item: item[metric],
            reverse=True,
        )