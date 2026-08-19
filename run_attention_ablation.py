"""
run_attention_ablation.py

EXP-003: Controlled attention ablation for HLI-01 v1.0.0.
"""

from run_experiment import initialize_experiment
from train import run_training


SEED = 42


def run_variant(name, use_attention):
    experiment_dir, logger = initialize_experiment(
        experiment_name=name,
        seed=SEED,
    )

    checkpoint_dir = experiment_dir / "checkpoints"
    figure_dir = experiment_dir / "figures"

    history, summary = run_training(
        checkpoint_dir=checkpoint_dir,
        figure_dir=figure_dir,
        seed=SEED,
        use_attention=use_attention,
    )

    logger.save_training_history(history)
    logger.save_metrics(summary)

    return experiment_dir, summary


def main():
    attention_dir, attention_summary = run_variant(
        name="exp003_attention_on",
        use_attention=True,
    )

    no_attention_dir, no_attention_summary = run_variant(
        name="exp003_attention_off",
        use_attention=False,
    )

    print()
    print("=" * 60)
    print("EXP-003 Attention Ablation Summary")
    print("=" * 60)

    print()
    print("Attention ON")
    print("Directory:", attention_dir)
    print(
        "Accuracy:",
        f"{attention_summary['test_accuracy']:.4f}",
    )
    print(
        "F1-score:",
        f"{attention_summary['test_f1_score']:.4f}",
    )

    print()
    print("Attention OFF")
    print("Directory:", no_attention_dir)
    print(
        "Accuracy:",
        f"{no_attention_summary['test_accuracy']:.4f}",
    )
    print(
        "F1-score:",
        f"{no_attention_summary['test_f1_score']:.4f}",
    )

    accuracy_delta = (
        attention_summary["test_accuracy"]
        - no_attention_summary["test_accuracy"]
    )

    f1_delta = (
        attention_summary["test_f1_score"]
        - no_attention_summary["test_f1_score"]
    )

    print()
    print(
        "Accuracy delta (ON - OFF):",
        f"{accuracy_delta:.4f}",
    )
    print(
        "F1 delta (ON - OFF):",
        f"{f1_delta:.4f}",
    )


if __name__ == "__main__":
    main()
