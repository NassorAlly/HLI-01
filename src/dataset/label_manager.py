"""
label_manager.py

Creates automatic mappings between class names and numeric labels.
"""


class LabelManager:
    """
    Handles encoding and decoding of dataset labels.
    """

    def __init__(self, class_names):
        """
        Parameters
        ----------
        class_names : list
            List of dataset class names.
        """

        self.class_names = sorted(class_names)

        self.label_to_id = {
            label: idx
            for idx, label in enumerate(self.class_names)
        }

        self.id_to_label = {
            idx: label
            for label, idx in self.label_to_id.items()
        }

    def encode(self, label):
        """
        Convert class name to numeric ID.
        """
        return self.label_to_id[label]

    def decode(self, label_id):
        """
        Convert numeric ID back to class name.
        """
        return self.id_to_label[label_id]

    def get_labels(self):
        """
        Return all labels.
        """
        return self.class_names

    def get_num_classes(self):
        """
        Return total number of classes.
        """
        return len(self.class_names)
