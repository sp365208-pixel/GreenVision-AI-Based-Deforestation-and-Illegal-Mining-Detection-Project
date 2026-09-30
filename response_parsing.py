import io
import base64
import numpy as np
from typing import List, Tuple
import requests
from PIL import Image
 
 
def base64_to_numpy(img_b64):
    img_bytes = io.BytesIO(base64.b64decode(img_b64))
    img_array = np.load(img_bytes)
    return img_array
 
 
def request_satellite_images(url, latitude, longitude, start_date, end_date, sample_number=2):
    if int(sample_number) > 2:
        endpoint = "get_interval_images"
    else:
        endpoint = "get_satellite_images"
 
    full_url = f"{url.rstrip('/')}/{endpoint}"
 
    if hasattr(start_date, 'strftime'):
        start_date = start_date.strftime("%Y-%m-%d")
    if hasattr(end_date, 'strftime'):
        end_date = end_date.strftime("%Y-%m-%d")
 
    response = requests.get(
        url=full_url,
        params={
            'start_timeframe': start_date,
            'end_timeframe': end_date,
            'longitude': longitude,
            'latitude': latitude,
            'sample_number': sample_number,
            'send_orginal_images': 'True'
        },
        timeout=120
    )
    response.raise_for_status()
    return response
 
 
def parse_response(response):
    data = response.json()
 
    backend_error = data.get("error")
    if backend_error:
        raise ValueError(f"Backend API Error: {backend_error}")
 
    image_dates = data.get("date_list_loaded", [])
    segmented_images_b64 = data.get("segmented_img_list", [])
    raw_images_b64 = data.get("original_img_list", [])
 
    if not segmented_images_b64 or not raw_images_b64 or not image_dates:
        raise ValueError("API response is missing necessary image or date data.")
 
    segmented_images = [base64_to_numpy(img_b64) for img_b64 in segmented_images_b64]
 
    raw_images = []
    for img_b64 in raw_images_b64:
        arr = base64_to_numpy(img_b64)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        img = Image.fromarray(arr, mode='RGB')
        raw_images.append(img)
 
    return image_dates, segmented_images, raw_images
 