import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_dim: int = 40, hidden_dim: int = 95, num_layers: int = 2, num_classes: int = 10, bidirectional: bool = False):
        super().__init__()
        self.is_cnn = False
        self.input_dim = input_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        classifier_in_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(classifier_in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() == 3 and x.size(1) == self.input_dim and x.size(2) != self.input_dim:
            x = x.transpose(1, 2)

        out, _ = self.gru(x)
        pooled = torch.mean(out, 1)
        return self.classifier(pooled)
