"""
=========================================================
HLI-01 Version 1.0.0
Visualization Dashboard
=========================================================
"""

import os
from pathlib import Path

from .visualization_utils import create_directory


class DashboardGenerator:
    """
    Generates an HTML dashboard for experiment visualization.

    Supports an optional experiment comparison figure while
    preserving compatibility with earlier HLI-01 versions.
    """

    def __init__(self, save_dir="outputs/dashboard"):
        self.save_dir = save_dir
        create_directory(self.save_dir)

    def generate(
        self,
        model_name,
        dataset_name,
        accuracy,
        macro_f1,
        filename="index.html",
        comparison_figure=None,
    ):
        filepath = os.path.join(
            self.save_dir,
            filename,
        )

        comparison_section = ""

        if comparison_figure is not None:
            comparison_figure = Path(
                comparison_figure
            )

            if not comparison_figure.exists():
                raise FileNotFoundError(
                    f"Comparison figure not found: "
                    f"{comparison_figure}"
                )

            comparison_section = f"""
<div class="card">

<h2>Experiment Comparison</h2>

<img src="{comparison_figure.name}"
     alt="Experiment Comparison">

</div>
"""

        html = f"""
<!DOCTYPE html>
<html>
<head>

<meta charset="utf-8">

<title>HLI-01 Dashboard</title>

<style>

body{{
font-family:Arial;
margin:40px;
background:#f4f6f9;
}}

h1{{
color:#003366;
}}

.card{{
background:white;
padding:20px;
margin-bottom:20px;
border-radius:10px;
box-shadow:0 2px 8px rgba(0,0,0,.15);
}}

img{{
max-width:100%;
border:1px solid #ddd;
margin-top:10px;
}}

table{{
width:100%;
border-collapse:collapse;
}}

td{{
padding:8px;
border-bottom:1px solid #ddd;
}}

</style>

</head>

<body>

<h1>HLI-01 Visualization Dashboard</h1>

<div class="card">

<h2>Experiment Information</h2>

<table>

<tr>
<td><b>Model</b></td>
<td>{model_name}</td>
</tr>

<tr>
<td><b>Dataset</b></td>
<td>{dataset_name}</td>
</tr>

<tr>
<td><b>Accuracy</b></td>
<td>{accuracy:.2f}%</td>
</tr>

<tr>
<td><b>Macro F1</b></td>
<td>{macro_f1:.2f}%</td>
</tr>

</table>

</div>

<div class="card">

<h2>Training Curves</h2>

<img src="../figures/loss_curve.png"
     alt="Loss Curve">

<img src="../figures/accuracy_curve.png"
     alt="Accuracy Curve">

</div>

<div class="card">

<h2>Confusion Matrix</h2>

<img src="../confusion_matrix/confusion_matrix.png"
     alt="Confusion Matrix">

</div>

<div class="card">

<h2>Classification Metrics</h2>

<img src="../metrics/classification_metrics.png"
     alt="Classification Metrics">

</div>

{comparison_section}

<div class="card">

<h2>Prediction Summary</h2>

<iframe
src="../predictions/prediction_summary.txt"
width="100%"
height="160">
</iframe>

</div>

<div class="card">

<h2>Experiment Report</h2>

<a href="../reports/experiment_report.pdf">
Open PDF Report
</a>

</div>

</body>
</html>
"""

        with open(
            filepath,
            "w",
            encoding="utf-8",
        ) as file:
            file.write(html)

        return filepath
