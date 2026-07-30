"""
Model Factory.

Creates deep learning models.
"""

from .lstm_model import LSTMModel
from .transformer_model import TransformerModel


class ModelFactory:

    @staticmethod
    def create(
        model_name,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.3,
    ):

        model_name = model_name.lower()

        if model_name == "lstm":

            return LSTMModel(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_classes=num_classes,
                dropout=dropout,
            )

        elif model_name == "transformer":

            return TransformerModel()

        else:

            raise ValueError(
                f"Unknown model: {model_name}"
            )
