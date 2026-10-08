import torch
import torch.nn as nn


class BiRNN(nn.Module):
    def __init__(self, input_dim: int = 40, hidden_dim: int = 64, num_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.is_cnn = False
        self.input_dim = input_dim

        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            nonlinearity='tanh'
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() == 3 and x.size(1) == self.input_dim and x.size(2) != self.input_dim:
            x = x.transpose(1, 2)

        out, _ = self.rnn(x)  # out: (B, T, hidden_dim * 2)

        # Temporal mean pooling
        pooled = torch.mean(out, dim=1)
        return self.classifier(pooled)
