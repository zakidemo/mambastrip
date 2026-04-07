import torch
import torch.nn as nn

class StripDecoder(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.Sigmoid()  # 🔥 مهم باش يرجع [0,1]
        )

    def forward(self, x, C, H, W):
        """
        x: [N, embed_dim]
        return: [N, C, H, W]
        """
        N, D = x.shape

        out = self.proj(x)          # [N, C*H*W]
        out = out.view(N, C, H, W)  # reshape
        
        return out