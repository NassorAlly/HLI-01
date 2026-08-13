"""
Transformer model placeholder.

The Transformer architecture is reserved for future development and is not implemented in Version 1.0.0.
"""

from .base_model import BaseModel


class TransformerModel(BaseModel):

    def __init__(self):

        super().__init__()

    def forward(self, x):

        raise NotImplementedError(
            "Transformer model is not implemented yet."
        )

    def get_model_name(self):

        return "Transformer"
