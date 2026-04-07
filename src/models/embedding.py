import torch
import torch.nn as nn

class StripEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, strips):
        """
        strips: [N, C, H, W]
        """
        N, C, H, W = strips.shape
        
        # flatten
        strips = strips.view(N, -1)  # [N, C*H*W]
        
        # projection
        out = self.proj(strips)      # [N, embed_dim]
        
        return out