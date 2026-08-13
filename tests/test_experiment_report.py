"""
=========================================================
HLI-01 Version 0.7.0
Testing Experiment Report Generator
=========================================================
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.visualization.experiment_report import ExperimentReportGenerator


def test_experiment_report():

    print("=" * 60)
    print("HLI-01 Version 0.7.0")
    print("Testing Experiment Report Generator")
    print("=" * 60)

    prediction = {
        "ground_truth": "peace",
        "prediction": "peace",
        "confidence": 99.31,
        "status": "✓ Correct",
    }

    model_name = "BiLSTM + Attention"
    dataset_name = "HLI-01 Dataset"
    num_classes = 4
    accuracy = 98.20
    macro_f1 = 97.80

    assert prediction["ground_truth"], (
        "Ground-truth label must not be empty."
    )

    assert prediction["prediction"], (
        "Prediction label must not be empty."
    )

    assert 0.0 <= prediction["confidence"] <= 100.0, (
        "Prediction confidence must be between 0 and 100."
    )

    assert num_classes > 0, (
        "Number of classes must be greater than zero."
    )

    assert 0.0 <= accuracy <= 100.0, (
        "Accuracy must be between 0 and 100."
    )

    assert 0.0 <= macro_f1 <= 100.0, (
        "Macro F1 must be between 0 and 100."
    )

    generator = ExperimentReportGenerator()

    report = generator.generate(
        model_name=model_name,
        dataset_name=dataset_name,
        num_classes=num_classes,
        accuracy=accuracy,
        macro_f1=macro_f1,
        prediction_summary=prediction,
    )

    print()
    print("Report Saved:")
    print(report)

    assert report is not None, (
        "The report generator did not return an output path."
    )

    assert isinstance(report, (str, os.PathLike)), (
        "The returned report path has an invalid type."
    )

    assert os.path.exists(report), (
        "Experiment report was not generated."
    )

    assert os.path.isfile(report), (
        "The returned experiment report path is not a file."
    )

    assert os.path.getsize(report) > 0, (
        "The generated experiment report is empty."
    )

    assert str(report).lower().endswith(".pdf"), (
        "The experiment report must be generated as a PDF file."
    )

    print()
    print("✓ Experiment report generated")
    print()
    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_experiment_report()


def test_experiment_report_with_comparison_figure(tmp_path):
    import matplotlib.pyplot as plt

    comparison_path = tmp_path / "comparison.png"

    fig, ax = plt.subplots()
    ax.bar(
        ["LSTM", "BiLSTM"],
        [0.90, 0.95],
    )
    fig.savefig(comparison_path)
    plt.close(fig)

    generator = ExperimentReportGenerator(
        save_dir=tmp_path,
    )

    prediction = {
        "ground_truth": "peace",
        "prediction": "peace",
        "confidence": 99.31,
        "status": "✓ Correct",
    }

    report = generator.generate(
        model_name="BiLSTM + Attention",
        dataset_name="HLI-01 Dataset",
        num_classes=4,
        accuracy=98.20,
        macro_f1=97.80,
        prediction_summary=prediction,
        comparison_figure=comparison_path,
        filename="experiment_report_with_comparison.pdf",
    )

    assert os.path.exists(report)
    assert os.path.isfile(report)
    assert os.path.getsize(report) > 0


def test_experiment_report_rejects_missing_comparison_figure(tmp_path):
    generator = ExperimentReportGenerator(
        save_dir=tmp_path,
    )

    prediction = {
        "ground_truth": "peace",
        "prediction": "peace",
        "confidence": 99.31,
        "status": "✓ Correct",
    }

    missing_figure = tmp_path / "missing_comparison.png"

    try:
        generator.generate(
            model_name="BiLSTM + Attention",
            dataset_name="HLI-01 Dataset",
            num_classes=4,
            accuracy=98.20,
            macro_f1=97.80,
            prediction_summary=prediction,
            comparison_figure=missing_figure,
            filename="should_not_generate.pdf",
        )
    except FileNotFoundError as error:
        assert "Comparison figure not found" in str(error)
    else:
        raise AssertionError(
            "Expected FileNotFoundError for missing comparison figure."
        )
