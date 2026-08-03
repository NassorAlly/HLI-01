"""
=========================================================
HLI-01 Version 0.7.0
Testing Prediction Visualizer
=========================================================
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.visualization.prediction_visualizer import PredictionVisualizer


def test_prediction_visualizer():

    print("=" * 60)
    print("HLI-01 Version 0.7.0")
    print("Testing Prediction Visualizer")
    print("=" * 60)

    ground_truth = "peace"
    prediction = "peace"
    confidence = 99.31

    visualizer = PredictionVisualizer()

    summary = visualizer.create_summary(
        ground_truth=ground_truth,
        prediction=prediction,
        confidence=confidence,
    )

    assert summary is not None, (
        "Prediction summary was not created."
    )

    assert isinstance(summary, dict), (
        "Prediction summary must be returned as a dictionary."
    )

    required_keys = {
        "ground_truth",
        "prediction",
        "confidence",
        "status",
    }

    assert required_keys.issubset(summary.keys()), (
        "Prediction summary is missing required fields."
    )

    assert summary["ground_truth"] == ground_truth, (
        "Ground-truth value is incorrect."
    )

    assert summary["prediction"] == prediction, (
        "Prediction value is incorrect."
    )

    assert summary["confidence"] == confidence, (
        "Confidence value is incorrect."
    )

    assert summary["status"], (
        "Prediction status must not be empty."
    )

    visualizer.print_summary(summary)

    output_file = visualizer.save_summary(summary)

    print()
    print("Saved To:")
    print(output_file)

    assert output_file is not None, (
        "The visualizer did not return an output path."
    )

    assert isinstance(output_file, (str, os.PathLike)), (
        "The returned output path has an invalid type."
    )

    assert os.path.exists(output_file), (
        "Prediction summary was not generated."
    )

    assert os.path.isfile(output_file), (
        "The prediction summary output is not a file."
    )

    assert os.path.getsize(output_file) > 0, (
        "The generated prediction summary file is empty."
    )

    with open(output_file, "r", encoding="utf-8") as summary_file:
        saved_content = summary_file.read()

    assert saved_content.strip(), (
        "The saved prediction summary contains no content."
    )

    assert ground_truth in saved_content, (
        "Ground-truth label is missing from the saved summary."
    )

    assert prediction in saved_content, (
        "Prediction label is missing from the saved summary."
    )

    print()
    print("✓ Prediction summary generated")
    print()
    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_prediction_visualizer()
