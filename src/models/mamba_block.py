import torch
import torch.nn as nn

class SimpleSSM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.A = nn.Parameter(torch.randn(dim))
        self.B = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        state = torch.zeros_like(x[:, 0])
        outputs = []

        for t in range(x.shape[1]):
            state = self.A * state + self.B * x[:, t]
            outputs.append(state.unsqueeze(1))

        return torch.cat(outputs, dim=1)