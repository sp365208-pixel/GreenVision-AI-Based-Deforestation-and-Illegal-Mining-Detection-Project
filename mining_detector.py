"""
mining_detector.py
==================
Illegal Mining Detection using a Lightweight U-Net CNN
- Built with PyTorch
- Trains on spectral indices (no pre-trained weights needed)
- Detects bare soil, exposed earth, mining pits using SWIR + BSI features
- Output: (512, 512) probability map — high values = likely mining
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

MODEL_SAVE_PATH = os.path.join(
    os.path.dirname(__file__), "model_ressources", "mining_unet.pt"
)

# ─────────────────────────────────────────────
# Lightweight U-Net Architecture
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)


class MiningUNet(nn.Module):
    """
    Lightweight U-Net for mining pixel segmentation.
    Input:  (B, 6, H, W) — 6 spectral feature channels
    Output: (B, 1, H, W) — mining probability per pixel
    """
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(6, 16)
        self.enc2 = ConvBlock(16, 32)
        self.enc3 = ConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(64, 128)

        # Decoder
        self.up3   = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3  = ConvBlock(128, 64)
        self.up2   = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2  = ConvBlock(64, 32)
        self.up1   = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1  = ConvBlock(32, 16)

        # Output
        self.out   = nn.Conv2d(16, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        # Bottleneck
        b  = self.bottleneck(self.pool(e3))
        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.sigmoid(self.out(d1))


# ─────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────

def extract_mining_features(img_array):
    """
    Extract 6 spectral features sensitive to mining activity.
    Input: (512, 512, 4) float32 [0-1] — R, G, B, NIR channels
    Output: (6, 512, 512) float32 tensor
    """
    r   = img_array[:, :, 0].astype(np.float32)
    g   = img_array[:, :, 1].astype(np.float32)
    b   = img_array[:, :, 2].astype(np.float32)
    nir = img_array[:, :, 3].astype(np.float32)
    eps = 1e-8

    # 1. NDVI — low = no vegetation = possible mining
    ndvi = (nir - r) / (nir + r + eps)

    # 2. BSI — Bare Soil Index — high = exposed earth
    bsi = ((r + g) - (nir + b)) / ((r + g) + (nir + b) + eps)

    # 3. NDWI — water index (mining often near water)
    ndwi = (g - nir) / (g + nir + eps)

    # 4. Brightness — mining pits are often bright/reflective
    brightness = (r + g + b + nir) / 4.0

    # 5. Red/NIR ratio — disturbed land signature
    red_nir = r / (nir + eps)

    # 6. Green/Blue ratio — sediment in water near mines
    green_blue = g / (b + eps)

    features = np.stack([ndvi, bsi, ndwi, brightness, red_nir, green_blue], axis=0)
    features = np.clip(features, -2, 2)  # clip outliers
    return features.astype(np.float32)


# ─────────────────────────────────────────────
# Pseudo-label Generation (Weak Supervision)
# ─────────────────────────────────────────────

def generate_mining_labels(img_array):
    r   = img_array[:, :, 0].astype(np.float32)
    g   = img_array[:, :, 1].astype(np.float32)
    b   = img_array[:, :, 2].astype(np.float32)
    nir = img_array[:, :, 3].astype(np.float32)
    eps = 1e-8

    ndvi = (nir - r) / (nir + r + eps)
    brightness = (r + g + b) / 3.0
    
    # Dark disturbed earth + low vegetation = mining pit
    # Catches both dark coal mines AND bright sandy mines
    mining_mask = (
        (ndvi < 0.20) &           # very low vegetation
        (brightness > 0.02)        # not pure black shadow
    ).astype(np.float32)

    return mining_mask


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_unet(img_array, epochs=15):
    """
    Train U-Net on a single satellite image using pseudo-labels.
    Uses patch-based training for memory efficiency.
    """
    print("[Mining U-Net] Extracting features...")
    features = extract_mining_features(img_array)   # (6, 512, 512)
    labels   = generate_mining_labels(img_array)    # (512, 512)

    # Create patches: 64x64 crops
    patch_size = 64
    patches_X, patches_Y = [], []
    for i in range(0, 512 - patch_size, patch_size // 2):
        for j in range(0, 512 - patch_size, patch_size // 2):
            patch_x = features[:, i:i+patch_size, j:j+patch_size]
            patch_y = labels[i:i+patch_size, j:j+patch_size]
            patches_X.append(patch_x)
            patches_Y.append(patch_y)

    X = torch.tensor(np.array(patches_X), dtype=torch.float32)
    Y = torch.tensor(np.array(patches_Y), dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, Y)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    device = torch.device("cpu")
    model  = MiningUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    print(f"[Mining U-Net] Training on {len(patches_X)} patches for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"[Mining U-Net] Epoch {epoch+1}/{epochs} — Loss: {total_loss/len(loader):.4f}")

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("[Mining U-Net] ✅ Model trained and saved!")
    return model


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

def load_or_train_unet(img_array):
    """Load saved U-Net or train a new one."""
    model = MiningUNet()
    if os.path.exists(MODEL_SAVE_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location="cpu"))
            print("[Mining U-Net] ✅ Loaded saved model")
            return model
        except Exception:
            print("[Mining U-Net] Saved model incompatible, retraining...")
    return train_unet(img_array)


def detect_mining(img_array, threshold=0.4):
    """
    Run mining detection on a satellite image.

    Args:
        img_array: (512, 512, 4) float32 [0-1] — R, G, B, NIR
        threshold: probability threshold for mining classification

    Returns:
        (512, 512) float32 binary mask — 1.0=mining detected, 0.0=clean
    """
    if img_array is None or img_array.ndim != 3:
        return np.zeros((512, 512), dtype=np.float32)

    model = load_or_train_unet(img_array)
    model.eval()

    features = extract_mining_features(img_array)  # (6, 512, 512)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)  # (1,6,512,512)

    with torch.no_grad():
        prob_map = model(x).squeeze().numpy()  # (512, 512)

    mining_mask = (prob_map > threshold).astype(np.float32)

    # Smooth
    from scipy.ndimage import uniform_filter
    mining_mask = (prob_map > 0.15).astype(np.float32)

    return mining_mask, prob_map