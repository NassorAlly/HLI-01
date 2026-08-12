import platform
import sys
from datetime import datetime
from typing import Dict, Any


class ExperimentMetadata:
    """
    Generate metadata for HLI-01 research experiments.
    """

    @staticmethod
    def generate(
        experiment_id: str,
        experiment_name: str,
        project_version: str = "1.0.0",
    ) -> Dict[str, Any]:
        return {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "project_version": project_version,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        }