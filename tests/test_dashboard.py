"""
=========================================================
HLI-01 Version 0.7.0
Testing Dashboard Generator
=========================================================
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)

from src.visualization.dashboard import DashboardGenerator


def test_dashboard():

    print("=" * 60)
    print("HLI-01 Version 0.7.0")
    print("Testing Dashboard Generator")
    print("=" * 60)

    dashboard = DashboardGenerator()

    html = dashboard.generate(

        model_name="BiLSTM + Attention",

        dataset_name="HLI-01 Dataset",

        accuracy=98.20,

        macro_f1=97.80

    )

    print()

    print("Dashboard Saved:")

    print(html)

    assert os.path.exists(html), \
        "Dashboard was not generated."

    print()

    print("✓ Dashboard generated")

    print()

    print("=" * 60)
    print("TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_dashboard()



def test_dashboard_with_comparison_figure(tmp_path):
    comparison_path = tmp_path / "experiment_comparison.png"
    comparison_path.write_bytes(b"fake image content")

    dashboard = DashboardGenerator(
        save_dir=tmp_path,
    )

    html_path = dashboard.generate(
        model_name="BiLSTM + Attention",
        dataset_name="HLI-01 Dataset",
        accuracy=98.20,
        macro_f1=97.80,
        comparison_figure=comparison_path,
        filename="dashboard_with_comparison.html",
    )

    assert os.path.exists(html_path)

    html_content = open(
        html_path,
        "r",
        encoding="utf-8",
    ).read()

    assert "Experiment Comparison" in html_content
    assert "experiment_comparison.png" in html_content


def test_dashboard_rejects_missing_comparison_figure(tmp_path):
    dashboard = DashboardGenerator(
        save_dir=tmp_path,
    )

    missing_figure = tmp_path / "missing_comparison.png"

    try:
        dashboard.generate(
            model_name="BiLSTM + Attention",
            dataset_name="HLI-01 Dataset",
            accuracy=98.20,
            macro_f1=97.80,
            comparison_figure=missing_figure,
            filename="should_not_generate.html",
        )
    except FileNotFoundError as error:
        assert "Comparison figure not found" in str(error)
    else:
        raise AssertionError(
            "Expected FileNotFoundError for missing comparison figure."
        )
