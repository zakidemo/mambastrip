import torch
import torch.nn as nn

from PIL import Image
import numpy as np

from src.data.strip_extractor import (
    image_to_strips,
    strips_to_image,
    image_to_horizontal_strips
)
from src.models.embedding import StripEmbedding
from src.models.mamba_block import SimpleMamba
from src.models.bottleneck import Bottleneck
from src.models.decoder import StripDecoder
from torch.utils.data import DataLoader
from src.data.dataset import ImageDataset

import math


# =========================
# Metrics
# =========================
def compute_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).detach()
    mse_val = mse.item()
    if mse_val == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse_val))


# =========================
# JPEG Baseline
# =========================
def jpeg_compress(img_tensor, quality=50):
    img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pil_img = Image.fromarray(img)

    import io
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    compressed = Image.open(buffer)
    compressed = np.array(compressed) / 255.0

    return torch.tensor(compressed).permute(2, 0, 1).float()


# =========================
# Model Config
# =========================
EMBED_DIM = 128

embed = StripEmbedding(input_dim=3 * 64 * 8, embed_dim=EMBED_DIM)
mamba = SimpleMamba(dim=EMBED_DIM)
bottleneck = Bottleneck()
decoder = StripDecoder(embed_dim=EMBED_DIM, output_dim=3 * 64 * 8)


# =========================
# Training Setup
# =========================
criterion = nn.MSELoss()

params = (
    list(embed.parameters())
    + list(mamba.parameters())
    + list(decoder.parameters())
)

optimizer = torch.optim.Adam(params, lr=1e-3)


# =========================
# Dataset
# =========================
dataset = ImageDataset("data/images", image_size=64)
loader = DataLoader(dataset, batch_size=4, shuffle=True)


# =========================
# Training Loop
# =========================
for epoch in range(50):

    for batch in loader:

        optimizer.zero_grad()

        total_loss = 0
        total_psnr = 0

        for img in batch:

            # =========================
            # Dual Strip Extraction
            # =========================
            v_strips = image_to_strips(img, patch_width=8)

            h_strips = image_to_horizontal_strips(img, patch_height=8)
            h_strips = h_strips.permute(0, 1, 3, 2)

            strips = torch.cat([v_strips, h_strips], dim=0)

            # =========================
            # Forward
            # =========================
            x = embed(strips)
            x = mamba(x)
            x = bottleneck(x)

            # =========================
            # Decode (vertical only)
            # =========================
            recon_strips = decoder(x, 3, 64, 8)

            recon_strips_v = recon_strips[:len(v_strips)]
            recon_img = strips_to_image(recon_strips_v)

            # =========================
            # 🔥 Residual Learning (NEW)
            # =========================
            recon_img = recon_img + img

            # clamp
            recon_img = torch.clamp(recon_img, 0, 1)

            # =========================
            # Loss
            # =========================
            loss = criterion(recon_img, img)
            total_loss += loss

            # =========================
            # Metric
            # =========================
            psnr = compute_psnr(recon_img, img)
            total_psnr += psnr

        # =========================
        # Backprop
        # =========================
        total_loss.backward()
        optimizer.step()

        avg_loss = total_loss.item() / len(batch)
        avg_psnr = total_psnr / len(batch)

        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f}")

        # =========================
        # JPEG Comparison
        # =========================
        jpeg_img = jpeg_compress(batch[0])
        jpeg_psnr = compute_psnr(jpeg_img, batch[0])

        print(f"👉 Mamba: {avg_psnr:.2f} dB | JPEG: {jpeg_psnr:.2f} dB")