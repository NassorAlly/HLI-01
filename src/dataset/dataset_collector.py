"""
dataset_collector.py

Discovers all sign language samples in the dataset directory.
"""

from pathlib import Path


class DatasetCollector:
    """
    Collects dataset file paths without loading their contents.
    """

    def __init__(self, dataset_path="dataset"):
        self.dataset_path = Path(dataset_path)

    def get_classes(self):
        """
        Return a sorted list of class names.
        """
        return sorted(
            [
                folder.name
                for folder in self.dataset_path.iterdir()
                if folder.is_dir()
            ]
        )

    def get_num_classes(self):
        """
        Return the number of classes.
        """
        return len(self.get_classes())

    def collect(self):
        """
        Collect all dataset files.

        Returns
        -------
        list
            List of dictionaries containing label and file path.
        """
        samples = []

        for label in self.get_classes():

            class_folder = self.dataset_path / label

            for file in class_folder.iterdir():

                if file.is_file():

                    samples.append(
                        {
                            "label": label,
                            "path": str(file)
                        }
                    )

        return samples

    def get_num_samples(self):
        """
        Return the total number of samples.
        """
        return len(self.collect())
