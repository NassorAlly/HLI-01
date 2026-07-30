"""
BiLSTM + Attention model.
"""

import torch
import torch.nn as nn

from .base_model import BaseModel


class Attention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.attention = nn.Linear(hidden_size * 2, 1)

    def forward(self, outputs):

        weights = torch.softmax(
            self.attention(outputs),
            dim=1
        )

        context = torch.sum(weights * outputs, dim=1)

        return context


class LSTMModel(BaseModel):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        dropout=0.3,
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

        self.attention = Attention(hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, x):

        outputs, _ = self.lstm(x)

        context = self.attention(outputs)

        context = self.dropout(context)

        return self.classifier(context)

    def get_model_name(self):

        return "BiLSTM_Attention"
