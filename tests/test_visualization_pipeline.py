"""
=========================================================
HLI-01 Version 0.6.0
Visualization Pipeline Integration Test
=========================================================
"""

import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from src.visualization.training_plots import TrainingPlotter
from src.visualization.confusion_matrix_plot import ConfusionMatrixPlotter
from src.visualization.metrics_plot import MetricsPlotter
from src.visualization.prediction_visualizer import PredictionVisualizer
from src.visualization.experiment_report import ExperimentReportGenerator
from src.visualization.dashboard import DashboardGenerator


def test_visualization_pipeline():

    print("=" * 70)
    print("HLI-01 Version 0.6.0")
    print("Visualization Pipeline Integration Test")
    print("=" * 70)

    # ==================================================
    # 1. Training Visualization
    # ==================================================

    print("\n[1/6] Training Visualization")

    trainer = TrainingPlotter()

    train_loss = [1.20, 0.95, 0.72, 0.55, 0.40]
    val_loss = [1.30, 1.05, 0.83, 0.66, 0.50]

    train_accuracy = [40, 58, 73, 86, 94]
    val_accuracy = [37, 55, 70, 82, 91]

    loss_file = trainer.plot_loss(
        train_loss,
        val_loss
    )

    accuracy_file = trainer.plot_accuracy(
        train_accuracy,
        val_accuracy
    )

    assert os.path.exists(loss_file), \
        "Loss curve not generated."

    assert os.path.exists(accuracy_file), \
        "Accuracy curve not generated."

    print("✓ Training plots generated")

    # ==================================================
    # 2. Confusion Matrix
    # ==================================================

    print("\n[2/6] Confusion Matrix")

    classes = [
        "hello",
        "no",
        "peace",
        "yes"
    ]

    cm = np.array([
        [96, 2, 1, 1],
        [3, 94, 2, 1],
        [0, 2, 97, 1],
        [1, 1, 2, 96]
    ])

    confusion = ConfusionMatrixPlotter()

    confusion_file = confusion.plot(
        cm,
        classes
    )

    assert os.path.exists(confusion_file), \
        "Confusion matrix not generated."

    print("✓ Confusion matrix generated")

    # ==================================================
    # 3. Metrics Visualization
    # ==================================================

    print("\n[3/6] Classification Metrics")

    metrics = {

        "precision": [
            0.98,
            0.95,
            0.99,
            0.97
        ],

        "recall": [
            0.97,
            0.94,
            0.98,
            0.96
        ],

        "f1_score": [
            0.975,
            0.945,
            0.985,
            0.965
        ],

        "support": [
            100,
            100,
            100,
            100
        ]

    }

    metric_plotter = MetricsPlotter()

    metrics_file = metric_plotter.plot(
        metrics,
        classes
    )

    assert os.path.exists(metrics_file), \
        "Metrics plot not generated."

    print("✓ Metrics visualization generated")

    # ==================================================
    # 4. Prediction Visualization
    # ==================================================

    print("\n[4/6] Prediction Summary")

    predictor = PredictionVisualizer()

    summary = predictor.create_summary(

        ground_truth="peace",

        prediction="peace",

        confidence=99.31

    )

    summary_file = predictor.save_summary(summary)

    assert os.path.exists(summary_file), \
        "Prediction summary not generated."

    print("✓ Prediction summary generated")

    # ==================================================
    # 5. Experiment Report
    # ==================================================

    print("\n[5/6] Experiment Report")

    report = ExperimentReportGenerator()

    report_file = report.generate(

        model_name="BiLSTM + Attention",

        dataset_name="HLI-01 Dataset",

        num_classes=4,

        accuracy=98.20,

        macro_f1=97.80,

        prediction_summary=summary

    )

    assert os.path.exists(report_file), \
        "Experiment report not generated."

    print("✓ Experiment report generated")

    # ==================================================
    # 6. Dashboard
    # ==================================================

    print("\n[6/6] Visualization Dashboard")

    dashboard = DashboardGenerator()

    dashboard_file = dashboard.generate(

        model_name="BiLSTM + Attention",

        dataset_name="HLI-01 Dataset",

        accuracy=98.20,

        macro_f1=97.80

    )

    assert os.path.exists(dashboard_file), \
        "Dashboard not generated."

    print("✓ Dashboard generated")

    # ==================================================
    # Final Summary
    # ==================================================

    print()
    print("=" * 70)
    print("HLI-01 Version 0.6.0")
    print("ALL VISUALIZATION MODULES PASSED")
    print("=" * 70)

    print()

    print("Generated Files")

    print("------------------------------")

    print(f"Loss Curve          : {loss_file}")
    print(f"Accuracy Curve      : {accuracy_file}")
    print(f"Confusion Matrix    : {confusion_file}")
    print(f"Metrics Plot        : {metrics_file}")
    print(f"Prediction Summary  : {summary_file}")
    print(f"Experiment Report   : {report_file}")
    print(f"Dashboard           : {dashboard_file}")

    print()

    print("=" * 70)
    print("VISUALIZATION PIPELINE PASSED")
    print("=" * 70)


if __name__ == "__main__":
    test_visualization_pipeline()
