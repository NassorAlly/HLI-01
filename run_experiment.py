# ==========================================================
# HLI-01 Version 1.0.0
# Research Experiment Runner
# ==========================================================

from experiments.experiment_logger import ExperimentLogger
from experiments.experiment_manager import ExperimentManager
from src.utils.reproducibility import set_reproducibility
from train import run_training


def initialize_experiment(
    experiment_name: str = "baseline",
    seed: int = 42,
):
    """
    Initialize a research-ready HLI-01 experiment.

    This function:
    1. Applies reproducibility settings.
    2. Creates the experiment directory.
    3. Creates an experiment logger.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.

    seed : int
        Random seed used for reproducibility.

    Returns
    -------
    experiment_dir
        Directory created for the experiment.

    logger
        ExperimentLogger instance associated with the experiment.
    """

    # ------------------------------------------------------
    # 1. Reproducibility
    # ------------------------------------------------------
    set_reproducibility(
        seed=seed,
        deterministic=True,
    )

    # ------------------------------------------------------
    # 2. Experiment directory
    # ------------------------------------------------------
    manager = ExperimentManager()

    experiment_dir = manager.create_experiment(
        name=experiment_name,
    )

    # ------------------------------------------------------
    # 3. Experiment logger
    # ------------------------------------------------------
    logger = ExperimentLogger(
        experiment_dir=experiment_dir,
    )

    return experiment_dir, logger


def run_experiment(
    experiment_name: str = "baseline",
    seed: int = 42,
):
    """
    Run a complete research-ready HLI-01 experiment.

    The experiment runner initializes the experiment workspace,
    executes training, and stores research outputs.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment.

    seed : int
        Random seed used for reproducibility.

    Returns
    -------
    experiment_dir
        Directory created for the experiment.

    logger
        ExperimentLogger associated with the experiment.

    history : dict
        Training history returned by run_training().

    summary : dict
        Serializable training summary returned by run_training().
    """

    # ------------------------------------------------------
    # 1. Initialize experiment
    # ------------------------------------------------------
    experiment_dir, logger = initialize_experiment(
        experiment_name=experiment_name,
        seed=seed,
    )

    # ------------------------------------------------------
    # 2. Experiment-specific output directories
    # ------------------------------------------------------
    checkpoint_dir = experiment_dir / "checkpoints"
    figure_dir = experiment_dir / "figures"

    # ------------------------------------------------------
    # 3. Run training pipeline
    # ------------------------------------------------------
    history, summary = run_training(
        checkpoint_dir=checkpoint_dir,
        figure_dir=figure_dir,
        seed=seed,
    )

    # ------------------------------------------------------
    # 4. Save experiment results
    # ------------------------------------------------------
    logger.save_training_history(history)
    logger.save_metrics(summary)

    return experiment_dir, logger, history, summary


def main():
    """
    Run the default HLI-01 research experiment.
    """

    experiment_dir, _, _, summary = run_experiment(
        experiment_name="baseline",
        seed=42,
    )

    print()
    print("=" * 60)
    print("HLI-01 experiment completed successfully.")
    print("=" * 60)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Best epoch: {summary['best_epoch']}")
    print(
        "Best validation loss: "
        f"{summary['best_validation_loss']:.4f}"
    )


if __name__ == "__main__":
    main()
