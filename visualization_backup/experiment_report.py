"""
=========================================================
HLI-01 Version 0.6.0
Experiment Report Generator
=========================================================
"""

import os
from datetime import datetime

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from .visualization_utils import create_directory


class ExperimentReportGenerator:
    """
    Generates a PDF experiment report.
    """

    def __init__(self,
                 save_dir="outputs/reports"):

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
            filename="experiment_report.pdf"):

        filepath = os.path.join(
            self.save_dir,
            filename
        )

        doc = SimpleDocTemplate(filepath)

        styles = getSampleStyleSheet()

        story = []

        story.append(
            Paragraph(
                "<b>HLI-01 Experiment Report</b>",
                styles["Title"]
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                f"<b>Model:</b> {model_name}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Dataset:</b> {dataset_name}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Classes:</b> {num_classes}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Accuracy:</b> {accuracy:.2f}%",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"<b>Macro F1:</b> {macro_f1:.2f}%",
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                "<b>Prediction Summary</b>",
                styles["Heading2"]
            )
        )

        story.append(
            Paragraph(
                f"Ground Truth: {prediction_summary['ground_truth']}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"Prediction: {prediction_summary['prediction']}",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"Confidence: {prediction_summary['confidence']:.2f}%",
                styles["Normal"]
            )
        )

        story.append(
            Paragraph(
                f"Status: {prediction_summary['status']}",
                styles["Normal"]
            )
        )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                f"<b>Date:</b> {datetime.now()}",
                styles["Normal"]
            )
        )

        doc.build(story)

        return filepath
