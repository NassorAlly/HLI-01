"""
=========================================================
HLI-01 Version 0.7.0
Prediction Visualizer
=========================================================
"""

import os

from .visualization_utils import create_directory


class PredictionVisualizer:
    """
    Creates prediction summaries.
    """

    def __init__(self,
                 save_dir="outputs/predictions"):

        self.save_dir = save_dir

        create_directory(self.save_dir)

    def create_summary(
            self,
            ground_truth,
            prediction,
            confidence):

        correct = ground_truth == prediction

        status = "✓ Correct" if correct else "✗ Incorrect"

        summary = {

            "ground_truth": ground_truth,

            "prediction": prediction,

            "confidence": confidence,

            "status": status

        }

        return summary

    def print_summary(self,
                      summary):

        print("=" * 60)

        print("Prediction Summary")

        print("=" * 60)

        print(f"Ground Truth : {summary['ground_truth']}")

        print(f"Prediction   : {summary['prediction']}")

        print(
            f"Confidence   : {summary['confidence']:.2f}%"
        )

        print(f"Status       : {summary['status']}")

        print("=" * 60)

    def save_summary(
            self,
            summary,
            filename="prediction_summary.txt"):

        filepath = os.path.join(
            self.save_dir,
            filename
        )

        with open(filepath,
                  "w",
                  encoding="utf-8") as file:

            file.write("=" * 60 + "\n")

            file.write("Prediction Summary\n")

            file.write("=" * 60 + "\n")

            file.write(
                f"Ground Truth : {summary['ground_truth']}\n"
            )

            file.write(
                f"Prediction   : {summary['prediction']}\n"
            )

            file.write(
                f"Confidence   : {summary['confidence']:.2f}%\n"
            )

            file.write(
                f"Status       : {summary['status']}\n"
            )

        return filepath

