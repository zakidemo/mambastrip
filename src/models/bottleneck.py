import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        x: [L, D]
        """
        # quantization (تقريب)
        x_quant = torch.round(x)

        return x_quant