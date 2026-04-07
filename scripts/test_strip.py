import torch
from src.data.strip_extractor import image_to_strips

img = torch.randn(3, 64, 64)

strips = image_to_strips(img, patch_width=8)

print(strips.shape)