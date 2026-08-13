"""
=========================================================
HLI-01 Version 1.0.0
Experiment Report Generator
=========================================================
"""

import os
from datetime import datetime
from pathlib import Path

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
)

from .visualization_utils import create_directory


class ExperimentReportGenerator:
    """
    Generates a PDF experiment report.

    Supports optional experiment-comparison figures while
    preserving compatibility with earlier HLI-01 versions.
    """

    def __init__(self, save_dir="outputs/reports"):
        self.save_dir = save_dir
        create_directory(self.save_dir)

    def generate(
        self,
        model_name,
        dataset_name,
        num_classes,
        accuracy,
        macro_f1,
        prediction_summary,
        filename="experiment_report.pdf",
        comparison_figure=None,
    ):
        filepath = os.path.join(
            self.save_dir,
            filename,
        )

        doc = SimpleDocTemplate(filepath)

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "<b>HLI-01 Experiment Report</b>",
                styles["Title"],
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                f"<b>Model:</b> {model_name}",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"<b>Dataset:</b> {dataset_name}",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"<b>Classes:</b> {num_classes}",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"<b>Accuracy:</b> {accuracy:.2f}%",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"<b>Macro F1:</b> {macro_f1:.2f}%",
                styles["Normal"],
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                "<b>Prediction Summary</b>",
                styles["Heading2"],
            )
        )

        story.append(
            Paragraph(
                f"Ground Truth: {prediction_summary['ground_truth']}",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"Prediction: {prediction_summary['prediction']}",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"Confidence: {prediction_summary['confidence']:.2f}%",
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                f"Status: {prediction_summary['status']}",
                styles["Normal"],
            )
        )

        if comparison_figure is not None:
            comparison_figure = Path(comparison_figure)

            if not comparison_figure.exists():
                raise FileNotFoundError(
                    f"Comparison figure not found: "
                    f"{comparison_figure}"
                )

            story.append(Spacer(1, 18))

            story.append(
                Paragraph(
                    "<b>Experiment Comparison</b>",
                    styles["Heading2"],
                )
            )

            story.append(Spacer(1, 8))

            image_reader = ImageReader(
                str(comparison_figure)
            )

            image_width, image_height = (
                image_reader.getSize()
            )

            max_width = 450

            scale = min(
                1.0,
                max_width / image_width,
            )

            story.append(
                Image(
                    str(comparison_figure),
                    width=image_width * scale,
                    height=image_height * scale,
                )
            )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                f"<b>Date:</b> {datetime.now()}",
                styles["Normal"],
            )
        )

        doc.build(story)

        return filepath
