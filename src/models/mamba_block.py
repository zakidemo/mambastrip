import torch
import torch.nn as nn

class SimpleMamba(nn.Module):
    def __init__(self, dim):
        super().__init__()
        
        self.A = nn.Parameter(torch.randn(dim))
        self.B = nn.Parameter(torch.randn(dim))
        self.C = nn.Parameter(torch.randn(dim))

    def forward(self, x):
        """
        x: [L, D]
        """
        L, D = x.shape
        
        state = torch.zeros(D)
        outputs = []

        for t in range(L):
            state = self.A * state + self.B * x[t]
            
            # non-linearity added 🔥
            y = torch.tanh(self.C * state)
            
            outputs.append(y.unsqueeze(0))

        return torch.cat(outputs, dim=0)