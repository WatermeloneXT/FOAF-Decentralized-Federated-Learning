import torch
import torch.nn as nn

class GlobalAverage(nn.Module):
    def __init__(self, data: torch.Tensor):
        super().__init__()
        self.data = data
        self.avg = data[:,2].mean(dtype=float)
        self.p = nn.Embedding(1, 1)

    def forward(self, users, items):
        preds = torch.ones_like(items) * self.avg
        return preds.unsqueeze(-1)
