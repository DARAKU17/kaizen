import torch
import torch.nn as nn


# =========================
# ATTENTION
# =========================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.attn = nn.Linear(
            hidden_dim * 2,
            1
        )

    def forward(self, x):
        """
        x: (B, T, 256)
        """

        scores = self.attn(x)              # (B,T,1)
        weights = torch.softmax(
            scores,
            dim=1
        )

        context = torch.sum(
            weights * x,
            dim=1
        )                                  # (B,256)

        return context


# =========================
# MODEL
# =========================
class WakeCRNN(nn.Module):
    def __init__(self):
        super().__init__()

        # input = (B,1,40,200)
        self.cnn = nn.Sequential(

            nn.Conv2d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # -> (16,20,100)

            nn.Conv2d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),   # -> (32,10,50)
        )

        # 32 * 10 = 320
        self.rnn = nn.GRU(
            input_size=320,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.attention = Attention(128)

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            256,
            1
        )

    def forward(self, x):
        """
        x: (B,1,40,200)
        """

        # CNN
        x = self.cnn(x)

        # (B,C,F,T)
        B, C, F, T = x.shape

        # -> (B,T,C,F)
        x = x.permute(0, 3, 1, 2)

        # -> (B,T,320)
        x = x.reshape(
            B,
            T,
            C * F
        )

        # GRU
        x, _ = self.rnn(x)

        # attention
        x = self.attention(x)

        x = self.dropout(x)

        # raw logits
        x = self.fc(x)

        return x