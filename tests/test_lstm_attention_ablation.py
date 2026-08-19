import torch

from src.models.lstm_model import LSTMModel


def test_lstm_without_attention_forward_shape():
    model = LSTMModel(
        input_size=63,
        hidden_size=128,
        num_layers=2,
        num_classes=4,
        use_attention=False,
    )

    dummy_input = torch.randn(8, 30, 63)

    output = model(dummy_input)

    assert output.shape == (8, 4)
