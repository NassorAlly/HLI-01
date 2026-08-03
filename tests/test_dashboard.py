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

