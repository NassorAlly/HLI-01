import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from experiments.metadata import ExperimentMetadata
from src.utils.config_loader import ConfigLoader


class ExperimentManager:
    """
    Manage research experiment directories for HLI-01.
    """

    def __init__(
        self,
        base_dir: str = "outputs/experiments",
        config_dir: str = "configs",
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.config_loader = ConfigLoader(config_dir=config_dir)

    def create_experiment(self, name: Optional[str] = None) -> Path:
        """
        Create a new experiment directory with metadata
        and a configuration snapshot.
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if name:
            safe_name = name.strip().replace(" ", "_")
            experiment_name = f"EXP_{timestamp}_{safe_name}"
        else:
            safe_name = "unnamed"
            experiment_name = f"EXP_{timestamp}"

        experiment_dir = self.base_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=False)

        # Create research output directories
        (experiment_dir / "figures").mkdir()
        (experiment_dir / "predictions").mkdir()
        (experiment_dir / "checkpoints").mkdir()

        # --------------------------------------------------
        # Metadata
        # --------------------------------------------------
        metadata = ExperimentMetadata.generate(
            experiment_id=experiment_name,
            experiment_name=safe_name,
            project_version="1.0.0",
        )

        metadata_path = experiment_dir / "metadata.json"

        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=4)

        # --------------------------------------------------
        # Configuration snapshot
        # --------------------------------------------------
        config_snapshot = self.config_loader.load_all()

        config_path = experiment_dir / "config.json"

        with open(config_path, "w", encoding="utf-8") as file:
            json.dump(config_snapshot, file, indent=4)

        return experiment_dir