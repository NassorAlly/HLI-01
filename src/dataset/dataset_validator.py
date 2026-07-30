"""
dataset_validator.py

Validates the integrity of the sign language dataset.
"""

import numpy as np

from .dataset_collector import DatasetCollector
from src.config.settings import (
    DATASET_PATH,
    SEQUENCE_LENGTH,
    NUM_FEATURES,
)


class DatasetValidator:
    """
    Validates all dataset samples.
    """

    def __init__(self, dataset_path=DATASET_PATH):
        self.collector = DatasetCollector(dataset_path)

    def validate(self):
        """
        Validate every sample in the dataset.

        Returns
        -------
        list
            List of validation errors.
        """

        samples = self.collector.collect()

        errors = []

        print("=" * 60)
        print("VALIDATING HLI-01 DATASET")
        print("=" * 60)

        for sample in samples:

            label = sample["label"]
            path = sample["path"]

            try:

                data = np.load(path)

                # Check shape
                if data.shape != (SEQUENCE_LENGTH, NUM_FEATURES):
                    errors.append(
                        f"{label}: {path} -> Invalid shape {data.shape}"
                    )

                # Check NaN
                if np.isnan(data).any():
                    errors.append(
                        f"{label}: {path} -> Contains NaN values"
                    )

                # Check Infinite values
                if np.isinf(data).any():
                    errors.append(
                        f"{label}: {path} -> Contains Infinite values"
                    )

            except Exception as e:

                errors.append(
                    f"{label}: {path} -> {str(e)}"
                )

        if len(errors) == 0:

            print("✅ Dataset validation PASSED")
            print(f"Checked {len(samples)} samples.")

        else:

            print("\n❌ Dataset validation FAILED\n")

            for error in errors:
                print(error)

        return errors
