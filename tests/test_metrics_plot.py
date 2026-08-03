"""
=========================================================
HLI-01 Version 0.7.0
Testing Metrics Plotter
=========================================================
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.visualization.metrics_plot import MetricsPlotter


def test_metrics_plot():

    print("=" * 60)
    print("HLI-01 Version 0.7.0")
    print("Testing Metrics Plotter")
    print("=" * 60)

    metrics = {
        "precision": [0.98, 0.95, 0.99, 0.97],
        "recall": [0.97, 0.94, 0.98, 0.96],
        "f1_score": [0.975, 0.945, 0.985, 0.965],
        "support": [100, 100, 100, 100],
    }

    class_names = [
        "hello",
        "no",
        "peace",
        "yes",
    ]

    expected_length = len(class_names)

    assert expected_length > 0, "Class names must not be empty."

    for metric_name, values in metrics.items():
        assert len(values) == expected_length, (
            f"Metric '{metric_name}' does not match the number of classes."
        )

    for metric_name in ("precision", "recall", "f1_score"):
        assert all(0.0 <= value <= 1.0 for value in metrics[metric_name]), (
            f"Metric '{metric_name}' contains values outside the range 0–1."
        )

    assert all(value >= 0 for value in metrics["support"]), (
        "Support values must not be negative."
    )

    plotter = MetricsPlotter()

    image = plotter.plot(
        metrics,
        class_names,
    )

    print()
    print("Metrics Figure Saved:")
    print(image)
    print()

    assert image is not None, (
        "The metrics plotter did not return an output path."
    )

    assert isinstance(image, (str, os.PathLike)), (
        "The returned metrics figure path has an invalid type."
    )

    assert os.path.exists(image), (
        "Metrics figure was not generated."
    )

    assert os.path.isfile(image), (
        "The returned metrics output is not a file."
    )

    assert os.path.getsize(image) > 0, (
        "The generated metrics figure is empty."
    )

    print("✓ Metrics figure generated")
    print()
    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_metrics_plot()
