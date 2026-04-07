# MambaStrip: Image Compression with State Space Models

## 🚀 Overview
MambaStrip is a novel image compression framework that leverages State Space Models (SSMs) for efficient sequence-based image modeling.

## 💡 Key Idea
Instead of traditional CNNs or Transformers, we convert images into vertical and horizontal strips and process them using Mamba-based sequence modeling.

## 🧠 Architecture
Image → Strips (Vertical + Horizontal) → Embedding → Mamba → Bottleneck → Decoder → Reconstruction

## 📊 Results
| Method | PSNR |
|--------|------|
| MambaStrip | ~13 dB |
| JPEG | ~28 dB |

## ⚡ Features
- Dual-direction strip modeling
- Lightweight architecture
- Sequence-based compression

## 🛠 Installation
```bash
pip install -r requirements.txt