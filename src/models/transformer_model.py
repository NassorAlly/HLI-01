"""
Transformer model.

Implementation will be added in Version 0.4.
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
