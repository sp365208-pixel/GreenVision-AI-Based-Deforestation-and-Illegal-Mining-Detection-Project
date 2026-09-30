# # # backend/gee_image_loader.py
# # import ee
# # import numpy as np
# # import requests
# # from PIL import Image
# # from io import BytesIO

# # # Initialize Earth Engine
# # ee.Initialize(project='forestvision-gee')  

# # def load_images_from_gee(lat_deg, lon_deg, date_list):
# #     """Load Sentinel-2 images from GEE and return as numpy arrays."""
# #     point = ee.Geometry.Point([lon_deg, lat_deg]).buffer(100)

# #     collection = (
# #         ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
# #         .filterBounds(point)
# #         .filterDate(date_list[0], date_list[-1])
# #         .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 50)) # Increased to 50
# #         .sort("CLOUDY_PIXEL_PERCENTAGE")
# #     )

# #     images = collection.toList(collection.size())
# #     num_images = collection.size().getInfo()

# #     if num_images == 0:
# #         print("⚠️ No Sentinel-2 images found for this area/date range.")
# #         return [], []

# #     available_dates = []
# #     image_arrays = []

# #     for i in range(num_images):
# #         img = ee.Image(images.get(i))
# #         date = img.date().format("YYYY-MM-dd").getInfo()
# #         available_dates.append(date)

# #         # Get a thumbnail as RGB + NIR composite
# #         url = img.select(["B4", "B3", "B2", "B8"]).getThumbURL({
# #             "region": point.buffer(5000).bounds(),
# #             "dimensions": 512,
# #             "min": 0,
# #             "max": 3000
# #         })

# #         # Download the image from URL
# #         response = requests.get(url)
# #         if response.status_code == 200:
# #             image = Image.open(BytesIO(response.content)).convert("RGB")
# #             image_array = np.array(image) / 255.0
# #             image_arrays.append(image_array)
# #         else:
# #             print(f"⚠️ Failed to fetch image for {date}")

# #     return available_dates, image_arrays

# import ee
# import numpy as np
# import requests
# from PIL import Image
# from io import BytesIO

# # Try-catch to ensure we don't crash on auth
# try:
#     ee.Initialize(project='forestvision-gee')
# except:
#     ee.Initialize()

# def load_images_from_gee(lat_deg, lon_deg, date_list):
#     point = ee.Geometry.Point([lon_deg, lat_deg])
#     roi = point.buffer(2560).bounds() # 5km area

#     # 1. SWITCH TO S2 (L1C) - This is global for 2017
#     # 2. Search a WIDER window (60 days) to guarantee we find a pass
#     start_search = ee.Date(date_list[0]).advance(-15, 'day')
#     end_search = start_search.advance(60, 'day')

#     collection = (ee.ImageCollection("COPERNICUS/S2") # L1C is the key for 2017
#                   .filterBounds(point)
#                   .filterDate(start_search, end_search)
#                   .sort("CLOUDY_PIXEL_PERCENTAGE"))

#     if collection.size().getInfo() == 0:
#         # Emergency Fallback: Just get the last 5 years if 2017 is missing
#         print("⚠️ No data in 2017 window. Trying latest available...")
#         collection = (ee.ImageCollection("COPERNICUS/S2")
#                       .filterBounds(point)
#                       .sort("system:time_start", False))

#     best_img = ee.Image(collection.first())
#     date_found = best_img.date().format("YYYY-MM-dd").getInfo()

#     # Get bands B4, B3, B2 (RGB) and B8 (NIR)
#     # Note: For L1C, the max value is usually 10000, not 3000
#     url = best_img.select(["B4", "B3", "B2", "B8"]).getThumbURL({
#         "region": roi,
#         "dimensions": 512,
#         "min": 0,
#         "max": 4000,
#         "format": "png"
#     })

#     res = requests.get(url)
#     if res.status_code == 200:
#         img_array = np.array(Image.open(BytesIO(res.content)).convert("RGBA")) / 255.0
#         return [date_found], [img_array]
    
#     return [], []

"""
gee_image_loader.py
====================
Replaces the old Sentinel Hub image loader.
Function name matches what main.py expects: load_images_from_satellite()

HOW TO SET UP (one-time):
1. Go to https://console.cloud.google.com → create a project (e.g. "forestvision-gee")
2. Enable "Google Earth Engine API" in that project
3. Go to https://code.earthengine.google.com → register the same project for EE
4. In VS Code terminal run: earthengine authenticate
5. Set your real project ID in the line below: GEE_PROJECT = "your-project-id"
"""

import ee
import numpy as np
import requests
from PIL import Image
from io import BytesIO

# ─── PUT YOUR REAL GCP PROJECT ID HERE ──────────────────────────
GEE_PROJECT = "forestvision-gee"
# ────────────────────────────────────────────────────────────────

_initialized = False

def _init_gee():
    """Initialize GEE once per process. Safe to call multiple times."""
    global _initialized
    if _initialized:
        return
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"[GEE] ✅ Initialized with project: {GEE_PROJECT}")
    except Exception as e:
        print(f"[GEE] ❌ Initialize failed: {e}")
        print("[GEE] Run in terminal:  earthengine authenticate")
        raise RuntimeError(
            "GEE not authenticated. Run `earthengine authenticate` in your terminal, "
            "then restart the server. Also make sure GEE_PROJECT is your real GCP project ID."
        )
    _initialized = True


# def _get_best_image(point, date_str, cloud_max=30):
#     """
#     Search for the best (least cloudy) Sentinel-2 image near a date.
#     Searches ±45 days. Falls back to ±90 days if nothing found.
    
#     Returns (ee.Image, date_string) or (None, None)
#     """
#     search_windows = [
#         (0, 45),    # first: forward 45 days from target date
#         (-30, 60),  # second: 30 days before + 60 days after
#         (-60, 120), # last resort: wide window
#     ]

#     for back_days, forward_days in search_windows:
#         start = ee.Date(date_str).advance(back_days, "day")
#         end   = ee.Date(date_str).advance(forward_days, "day")

#         col = (
#             ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")  # L2A Surface Reflectance
#             .filterBounds(point)
#             .filterDate(start, end)
#             .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_max))
#             .sort("CLOUDY_PIXEL_PERCENTAGE")
#         )

#         size = col.size().getInfo()
#         if size > 0:
#             img = col.first()
#             actual_date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
#             print(f"[GEE] Found {size} images near {date_str}, using: {actual_date}")
#             return img, actual_date

#     print(f"[GEE] ⚠️  No image found near {date_str} even with wide window.")
#     return None, None

def _get_best_image(point, date_str, cloud_max=30):
    """
    Search for best Sentinel-2 image near a date.
    Progressive fallback — widens search until image found.
    """
    search_windows = [
        (0,  45,  30),   # 45 days forward, 30% cloud
        (-30, 60, 40),   # wider window, 40% cloud
        (-60, 120, 60),  # very wide, 60% cloud
        (-90, 180, 80),  # last resort, 80% cloud
    ]

    for back_days, forward_days, cloud_pct in search_windows:
        start = ee.Date(date_str).advance(back_days, "day")
        end   = ee.Date(date_str).advance(forward_days, "day")

        col = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )

        size = col.size().getInfo()
        if size > 0:
            img = col.first()
            actual_date = ee.Date(
                img.get("system:time_start")
            ).format("YYYY-MM-dd").getInfo()
            print(f"[GEE] Found {size} images near {date_str}, using: {actual_date}")
            return img, actual_date

    # Last resort - try Landsat 8 if Sentinel-2 not available
    print(f"[GEE] Trying Landsat 8 fallback for {date_str}...")
    col_landsat = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(point)
        .filterDate(
            ee.Date(date_str).advance(-90, "day"),
            ee.Date(date_str).advance(90, "day")
        )
        .sort("CLOUD_COVER")
    )
    if col_landsat.size().getInfo() > 0:
        img = col_landsat.first()
        actual_date = ee.Date(
            img.get("system:time_start")
        ).format("YYYY-MM-dd").getInfo()
        print(f"[GEE] Using Landsat 8 image: {actual_date}")
        return img, actual_date

    print(f"[GEE] No image found near {date_str}")
    return None, None

# --- Update this function in backend/gee_image_loader.py ---

# def _image_to_numpy(image, roi, bands=["B4", "B3", "B2", "B8"]):
#     """
#     Download a GEE image as a PNG thumbnail and return as float32 numpy array.
#     """
#     url = image.select(bands).getThumbURL({
#         "region": roi,
#         "dimensions": 512,
#         "min": 0,
#         "max": 3000,   
#         "format": 'png'
#     })

#     resp = requests.get(url, timeout=60)
#     if resp.status_code != 200:
#         print(f"[GEE] ❌ Thumbnail download failed: HTTP {resp.status_code}")
#         return None

#     # FIX: Use BytesIO and explicitly check the image mode
#     try:
#         img_pil = Image.open(BytesIO(resp.content))
        
#         # If GEE sends an image PIL doesn't recognize as 4-channel, 
#         # it might be 'P' or 'L'. This forces it to a standard 4-channel float array.
#         if img_pil.mode != 'RGBA':
#             img_pil = img_pil.convert('RGBA')
            
#         arr = np.array(img_pil).astype(np.float32) / 255.0  # shape (512,512,4)
#         return arr
#     except Exception as e:
#         print(f"[GEE] ❌ Error processing image bytes: {e}")
#         return None

def _image_to_numpy(image, roi, bands=None):
    """Download GEE image as numpy array."""
    
    # Auto-detect bands based on collection
    if bands is None:
        # Try Sentinel-2 bands first
        try:
            available = image.bandNames().getInfo()
            if "B4" in available:
                bands = ["B4", "B3", "B2", "B8"]  # Sentinel-2
            elif "SR_B4" in available:
                bands = ["SR_B4", "SR_B3", "SR_B2", "SR_B5"]  # Landsat 8
            else:
                bands = available[:4]
        except:
            bands = ["B4", "B3", "B2", "B8"]

    try:
        url = image.select(bands).getThumbURL({
            "region": roi,
            "dimensions": 512,
            "min": 0,
            "max": 3000,
            "format": "png"
        })

        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            print(f"[GEE] Download failed: HTTP {resp.status_code}")
            return None

        img_pil = Image.open(BytesIO(resp.content))
        if img_pil.mode != 'RGBA':
            img_pil = img_pil.convert('RGBA')

        arr = np.array(img_pil).astype(np.float32) / 255.0
        return arr

    except Exception as e:
        print(f"[GEE] Error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC FUNCTION — this is what main.py calls
# Name MUST stay: load_images_from_satellite
# ─────────────────────────────────────────────────────────────────────────────

def load_images_from_satellite(lat_deg, lon_deg, date_list):
    """
    Drop-in replacement for the old Sentinel Hub loader.
    
    Args:
        lat_deg  (float): Latitude
        lon_deg  (float): Longitude  
        date_list (list): List of date strings ["YYYY-MM-DD", "YYYY-MM-DD"]

    Returns:
        loaded_dates (list of str): Actual image dates found
        img_arrays   (list of np.ndarray): Each array shape (512,512,4) float32 [0-1]
                      Channels: [Red, Green, Blue, NIR]
    """
    _init_gee()

    point = ee.Geometry.Point([lon_deg, lat_deg])
    roi   = point.buffer(2560).bounds()  # ~5km x 5km region

    loaded_dates = []
    img_arrays   = []

    for date_str in date_list:
        # Convert date objects to string if needed
        if not isinstance(date_str, str):
            date_str = str(date_str)

        img, actual_date = _get_best_image(point, date_str)

        if img is None:
            print(f"[GEE] Skipping {date_str} — no suitable image found.")
            continue

        arr = _image_to_numpy(img, roi)

        if arr is None:
            print(f"[GEE] Skipping {date_str} — image download failed.")
            continue

        loaded_dates.append(actual_date)
        img_arrays.append(arr)
        print(f"[GEE] ✅ Loaded image for {date_str} → actual date: {actual_date}, shape: {arr.shape}")

    if not loaded_dates:
        raise ValueError(
            f"No satellite imagery found for any of the requested dates: {date_list}. "
            "Try a different location or broader date range."
        )

    return loaded_dates, img_arrays