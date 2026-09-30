import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import base64
import datetime as dt
import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from backend.segmenter import segment
from backend.gee_image_loader import load_images_from_satellite
from backend.mining_detector import detect_mining

# ──────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────
def numpy_to_base64(img_array):
    img_bytes = io.BytesIO()
    np.save(img_bytes, img_array)
    img_bytes.seek(0)
    return base64.b64encode(img_bytes.read()).decode("utf-8")


def safe_response(error=None, dates=None, segmented=None, original=None):
    return JSONResponse(
        status_code=200,
        content={
            "error": error,
            "date_list_loaded": dates or [],
            "segmented_img_list": segmented or [],
            "original_img_list": original or []
        }
    )


print("✅ Forest: Random Forest Classifier")
print("✅ Mining: U-Net CNN (PyTorch)")
SEGMENTATION_MODEL = "rf+unet"


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────
app = FastAPI()


@app.get("/")
def index():
    return {
        "api_status": "running",
        "forest_model": "Random Forest Classifier",
        "mining_model": "U-Net CNN (PyTorch)",
        "data_source": "Google Earth Engine — Sentinel-2 L2A"
    }


# ──────────────────────────────────────────────────────────────
# Core Processing — Forest
# ──────────────────────────────────────────────────────────────
def process_images(latitude, longitude, date_list, send_original_images):
    loaded_dates, img_arrays = load_images_from_satellite(
        lat_deg=float(latitude),
        lon_deg=float(longitude),
        date_list=date_list
    )

    if not loaded_dates:
        raise ValueError("No suitable satellite data found for the selected timeframe.")

    original_img_arrays  = []
    segmented_img_arrays = []

    for img in img_arrays:
        vis = img[:, :, :3]
        vis_uint8 = (np.clip(vis, 0, 1) * 255).astype(np.uint8)
        original_img_arrays.append(vis_uint8)

        seg = segment(img, model=None, threshold=0.3)
        if seg is not None and seg.shape == (512, 512):
            segmented_img_arrays.append(seg)
        else:
            segmented_img_arrays.append(np.zeros((512, 512), dtype=np.float32))

    segmented_img_b64 = [numpy_to_base64(x) for x in segmented_img_arrays]
    original_img_b64  = [numpy_to_base64(x) for x in original_img_arrays]

    return loaded_dates, segmented_img_b64, original_img_b64


# ──────────────────────────────────────────────────────────────
# Forest Endpoints
# ──────────────────────────────────────────────────────────────
@app.get("/get_satellite_images")
def get_satellite_images(
    start_timeframe: str,
    end_timeframe: str,
    longitude: str,
    latitude: str,
    send_orginal_images: str = "False"
):
    try:
        dt.datetime.strptime(start_timeframe, "%Y-%m-%d")
        dt.datetime.strptime(end_timeframe, "%Y-%m-%d")
        date_list = [start_timeframe, end_timeframe]
        loaded_dates, segmented_img_b64, original_img_b64 = process_images(
            latitude, longitude, date_list, send_orginal_images
        )
        return safe_response(
            error=None,
            dates=loaded_dates,
            segmented=segmented_img_b64,
            original=original_img_b64 if send_orginal_images == "True" else []
        )
    except Exception as e:
        print(f"❌ Error in /get_satellite_images: {e}")
        import traceback; traceback.print_exc()
        return safe_response(error=str(e))


@app.get("/get_interval_images")
def get_interval_images(
    start_timeframe: str,
    end_timeframe: str,
    longitude: str,
    latitude: str,
    sample_number: str,
    send_orginal_images: str = "False"
):
    try:
        start_dt    = dt.datetime.strptime(start_timeframe, "%Y-%m-%d")
        end_dt      = dt.datetime.strptime(end_timeframe,   "%Y-%m-%d")
        num_samples = int(sample_number)
        if num_samples < 2:
            return safe_response(error="sample_number must be >= 2")
        delta     = (end_dt - start_dt) / (num_samples - 1)
        date_list = [
            (start_dt + i * delta).strftime("%Y-%m-%d")
            for i in range(num_samples)
        ]
        loaded_dates, segmented_img_b64, original_img_b64 = process_images(
            latitude, longitude, date_list, send_orginal_images
        )
        return safe_response(
            error=None,
            dates=loaded_dates,
            segmented=segmented_img_b64,
            original=original_img_b64 if send_orginal_images == "True" else []
        )
    except Exception as e:
        print(f"❌ Error in /get_interval_images: {e}")
        import traceback; traceback.print_exc()
        return safe_response(error=str(e))


# ──────────────────────────────────────────────────────────────
# Mining Detection Endpoint — U-Net CNN
# ──────────────────────────────────────────────────────────────
@app.get("/detect_mining")
def detect_mining_endpoint(
    latitude: str,
    longitude: str,
    date: str,
):
    """
    POST /detect_mining
    Runs U-Net mining detection on a single date image.
    Returns:
      - original satellite image (base64)
      - mining mask (base64) — binary 0/1
      - mining probability map (base64)
      - mining_percentage: float
      - actual_date: str
    """
    try:
        loaded_dates, img_arrays = load_images_from_satellite(
            lat_deg=float(latitude),
            lon_deg=float(longitude),
            date_list=[date]
        )

        if not loaded_dates or not img_arrays:
            return JSONResponse(status_code=200, content={
                "error": "No image found for this location and date."
            })

        img = img_arrays[0]
        actual_date = loaded_dates[0]

        # Run U-Net mining detection
        mining_mask, prob_map = detect_mining(img, threshold=0.4)

        # Original RGB image
        rgb = (np.clip(img[:, :, :3], 0, 1) * 255).astype(np.uint8)

        # Mining percentage
        mining_pct = float(mining_mask.sum() / mining_mask.size * 100)

        return JSONResponse(status_code=200, content={
            "error": None,
            "actual_date": actual_date,
            "mining_percentage": round(mining_pct, 2),
            "original_img": numpy_to_base64(rgb),
            "mining_mask":  numpy_to_base64(mining_mask),
            "prob_map":     numpy_to_base64(prob_map.astype(np.float32)),
        })

    except Exception as e:
        print(f"❌ Error in /detect_mining: {e}")
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=200, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)