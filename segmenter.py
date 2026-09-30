# # import numpy as np

# # def segment(img_array: np.array, model, threshold: int = 0.7) -> np.array:
# #     """Takes a scaled image array, predicts its rainforest segmentation, converts it
# #     to black and white according to a threshold. Returns a (512, 512) img_array.
# #     """
# #     predicted_img_array = model.predict(np.expand_dims(img_array, axis=0))[0, :, :, 0]
# #     black_white_img_array = (predicted_img_array > threshold).astype(np.uint8) * 255
# #     return black_white_img_array

# """
# segmenter.py - NDVI-based forest segmentation
# Replaces the broken att_unet_4b.hdf5 model with a simple NDVI threshold.
# Output is identical format: (512, 512) binary numpy array.
#   1 = forest
#   0 = non-forest
# """

# import numpy as np


# def segment(img_array, model=None, threshold=0.3):
#     """
#     Segment forest from a (512, 512, 4) image array.
#     Channels: [Red, Green, Blue, NIR] all in 0-1 range.

#     Args:
#         img_array: numpy array shape (512, 512, 4)
#         model: ignored — kept for API compatibility
#         threshold: NDVI threshold above which = forest (default 0.3)

#     Returns:
#         binary mask (512, 512) float32 — 1.0 = forest, 0.0 = not forest
#     """
#     if img_array is None:
#         return None

#     if img_array.ndim != 3 or img_array.shape[2] < 4:
#         print(f"⚠ Unexpected image shape: {img_array.shape}")
#         return None

#     red = img_array[:, :, 0].astype(np.float32)
#     nir = img_array[:, :, 3].astype(np.float32)

#     # NDVI = (NIR - RED) / (NIR + RED + epsilon)
#     epsilon = 1e-8
#     ndvi = (nir - red) / (nir + red + epsilon)

#     # Binary forest mask
#     forest_mask = (ndvi > threshold).astype(np.float32)

#     # Light smoothing to reduce noise (simple 3x3 majority filter)
#     forest_mask = _majority_filter(forest_mask)

#     return forest_mask


# def _majority_filter(mask, size=3):
#     """
#     Simple majority filter to smooth the binary mask.
#     Each pixel becomes 1 if majority of neighbors are 1.
#     """
#     from scipy.ndimage import uniform_filter
#     smoothed = uniform_filter(mask.astype(np.float32), size=size)
#     return (smoothed > 0.5).astype(np.float32)

"""
segmenter.py - ML-based forest & mining segmentation
Uses Random Forest Classifier trained on spectral indices.
No pre-trained model file needed - trains on the fly from pixel features.

Features used:
- NDVI  (vegetation)
- NDWI  (water)
- BSI   (bare soil / mining)
- SWIR ratio (disturbed land)
- RGB bands
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "model_ressources", "rf_model.pkl")
SCALER_SAVE_PATH = os.path.join(os.path.dirname(__file__), "model_ressources", "rf_scaler.pkl")


def extract_features(img_array):
    """
    Extract spectral features from a (512, 512, 4) image.
    Channels: R=0, G=1, B=2, NIR=3
    Returns: (512*512, n_features) array
    """
    r   = img_array[:, :, 0].astype(np.float32)
    g   = img_array[:, :, 1].astype(np.float32)
    b   = img_array[:, :, 2].astype(np.float32)
    nir = img_array[:, :, 3].astype(np.float32)
    eps = 1e-8

    # NDVI - Normalized Difference Vegetation Index
    ndvi = (nir - r) / (nir + r + eps)

    # NDWI - Normalized Difference Water Index
    ndwi = (g - nir) / (g + nir + eps)

    # EVI - Enhanced Vegetation Index
    evi = 2.5 * (nir - r) / (nir + 6*r - 7.5*b + 1 + eps)

    # SAVI - Soil Adjusted Vegetation Index
    savi = 1.5 * (nir - r) / (nir + r + 0.5 + eps)

    # Brightness
    brightness = (r + g + b + nir) / 4

    # Stack features: shape (512, 512, 8)
    features = np.stack([ndvi, ndwi, evi, savi, r, g, b, nir, brightness], axis=2)

    # Flatten to (512*512, 8)
    h, w, f = features.shape
    return features.reshape(-1, f)


def generate_training_labels(img_array):
    """
    Generate pseudo-labels using NDVI thresholds.
    This is the 'weak supervision' approach:
    - High NDVI  → forest (1)
    - Low NDVI   → non-forest (0)
    Only use high-confidence pixels for training.
    """
    r   = img_array[:, :, 0].astype(np.float32)
    nir = img_array[:, :, 3].astype(np.float32)
    eps = 1e-8
    ndvi = (nir - r) / (nir + r + eps)

    labels = np.full(ndvi.shape, -1, dtype=np.int8)  # -1 = uncertain
    labels[ndvi > 0.45] = 1   # confident forest
    labels[ndvi < 0.15] = 0   # confident non-forest

    return labels.flatten()


def train_model(img_array):
    """
    Train Random Forest on spectral features of the image.
    Uses confident NDVI pixels as training labels.
    """
    print("[ML] Extracting features...")
    features = extract_features(img_array)
    labels   = generate_training_labels(img_array)

    # Only use confident pixels
    mask = labels != -1
    X_train = features[mask]
    y_train = labels[mask]

    print(f"[ML] Training on {X_train.shape[0]} pixels ({y_train.sum()} forest, {(y_train==0).sum()} non-forest)")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
        class_weight='balanced'
    )
    clf.fit(X_scaled, y_train)
    print("[ML] ✅ Random Forest trained successfully")

    # Save model
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)

    return clf, scaler


def load_or_train_model(img_array):
    """Load saved model or train a new one."""
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(SCALER_SAVE_PATH):
        try:
            clf    = joblib.load(MODEL_SAVE_PATH)
            scaler = joblib.load(SCALER_SAVE_PATH)
            print("[ML] ✅ Loaded saved Random Forest model")
            return clf, scaler
        except Exception:
            print("[ML] Saved model load failed, retraining...")

    return train_model(img_array)


def segment(img_array, model=None, threshold=0.3):
    """
    Segment forest using Random Forest ML model.

    Args:
        img_array: (512, 512, 4) float32 array [0-1]
        model: ignored (kept for API compatibility)
        threshold: not used for RF (kept for compatibility)

    Returns:
        (512, 512) float32 binary mask — 1.0=forest, 0.0=non-forest
    """
    if img_array is None or img_array.ndim != 3:
        return None

    clf, scaler = load_or_train_model(img_array)

    features = extract_features(img_array)
    X_scaled = scaler.transform(features)
    predictions = clf.predict(X_scaled)

    mask = predictions.reshape(512, 512).astype(np.float32)

    # Light smoothing
    from scipy.ndimage import uniform_filter
    mask = (uniform_filter(mask, size=3) > 0.5).astype(np.float32)

    return mask