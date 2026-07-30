import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SignLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        last_output = output[:, -1, :]
        return self.fc(last_output)
