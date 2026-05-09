import torch
import torch.nn as nn

class KaizenWakeWord(nn.Module):
    def __init__(self, feature_size=120, hidden_size=256):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (time, batch, feature)

        out, _ = self.lstm(x)

        # take last timestep
        out = out[-1]  # (batch, hidden)

        return self.fc(out)