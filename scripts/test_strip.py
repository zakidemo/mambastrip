import torch
from src.data.strip_extractor import image_to_strips, strips_to_image
from src.models.embedding import StripEmbedding
from src.models.mamba_block import SimpleMamba
from src.models.bottleneck import Bottleneck
from src.models.decoder import StripDecoder

img = torch.randn(3, 64, 64)

# 1. strips
strips = image_to_strips(img, patch_width=8)

# 2. embedding
embed = StripEmbedding(input_dim=3*64*8, embed_dim=128)
x = embed(strips)

# 3. mamba
mamba = SimpleMamba(dim=128)
x = mamba(x)

# 4. bottleneck
bottleneck = Bottleneck()
x = bottleneck(x)

# 5. decoder
decoder = StripDecoder(embed_dim=128, output_dim=3*64*8)
recon_strips = decoder(x, C=3, H=64, W=8)

print("Strips shape:", recon_strips.shape)

# 6. reconstruction (🔴 هنا تحط الكود)
reconstructed_img = strips_to_image(recon_strips)

print("Final image shape:", reconstructed_img.shape)