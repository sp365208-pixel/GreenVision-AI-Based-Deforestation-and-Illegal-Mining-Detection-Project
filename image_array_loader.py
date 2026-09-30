# from typing import List
# from sentinelhub import SentinelHubDownloadClient
# from backend.sentinelhub_requester import (
#     search_optimal_l2a_tiles,
#     create_sentinelhub_token,
#     create_bounding_box,
#     build_sentinel_request
# )


# def load_images_from_satellite(
#         lat_deg: float,
#         lon_deg: float,
#         date_list: List[str]
#     ):
#     """Loads available image arrays from sentinel hub at multiple times."""
#     box = create_bounding_box(
#         lat_deg=lat_deg,
#         lon_deg=lon_deg,
#         image_size_px=512,
#         resolution_m_per_px=10
#     )
#     config, catalog_ = create_sentinelhub_token()
#     date_list_available = []
#     for date_i in date_list:
#         optimal_tile = search_optimal_l2a_tiles(
#             catalog=catalog_,
#             bbox=box,
#             date_request=date_i,
#             range_days=91,
#             max_cloud_coverage=10
#         )
#         if optimal_tile:
#            date_list_available.append(optimal_tile.get('date'))
#     list_of_requests_vis= [
#         build_sentinel_request(
#             config=config,
#             box=box,
#             request_type='TrueColor',
#             request_date=(date_i, date_i)
#         )
#         for date_i in date_list_available
#     ]
#     list_of_requests_fourband= [
#         build_sentinel_request(
#             config=config,
#             box=box,
#             request_type='4-band',
#             request_date=(date_i, date_i)
#         )
#         for date_i in date_list_available
#     ]
#     list_of_requests = list_of_requests_vis + list_of_requests_fourband
#     list_of_requests = [
#         request.download_list[0]
#         for request in list_of_requests
#     ]
#     img_arrays = SentinelHubDownloadClient(config=config).download(
#         list_of_requests,
#         max_threads=5
#     )
#     return date_list_available, img_arrays
import ee
import numpy as np
import requests
import cv2

# Initialize Earth Engine
try:
    ee.Initialize()
except:
    ee.Authenticate()
    ee.Initialize()


def download_image(image, region):
    # ... existing url code ...
    resp = requests.get(url)
    img_array = np.asarray(bytearray(resp.content), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # Force to 3-channel BGR
    # If you need 4 channels, keep IMREAD_UNCHANGED but add a mode check
    img = cv2.resize(img, (512, 512))
    return img

def load_images_from_satellite(lat_deg, lon_deg, date_list):

    point = ee.Geometry.Point([lon_deg, lat_deg])
    region = point.buffer(20000).bounds().getInfo()  # 20 km area

    end_date = ee.Date(date_list[-1]).advance(1, 'day')

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(point)
        .filterDate(date_list[0], end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 90))
        .sort('CLOUDY_PIXEL_PERCENTAGE')
    )

    total = collection.size().getInfo()
    print("GEE image count (cloud filter):", total)

    # Fallback if empty
    if total == 0:
        collection = (
           ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(point)
            .filterDate(date_list[0], end_date)
            .sort('system:time_start')
        )
        total = collection.size().getInfo()
        print("GEE image count (no cloud filter):", total)

    if total == 0:
        return [], []

    # Take best 3 images (not just 2)
    total = min(total, 3)

    image_list = collection.toList(total)

    vis_images = []
    nir_images = []
    loaded_dates = []

    for i in range(total):
        image = ee.Image(image_list.get(i))
        date_str = image.date().format("YYYY-MM-dd").getInfo()

        # RGB
        rgb = image.select(['B4', 'B3', 'B2']).divide(10000)
        rgb_np = download_image(rgb, region)
        rgb_np = rgb_np[:, :, :3] / 255.0

        # NIR
        nir = image.select(['B8']).divide(10000)
        nir_np = download_image(nir, region)
        nir_np = nir_np[:, :, 0] / 255.0

        nir_stack = np.zeros((512, 512, 4))
        nir_stack[:, :, 3] = nir_np

        vis_images.append(rgb_np)
        nir_images.append(nir_stack)
        loaded_dates.append(date_str)

    img_arrays = vis_images + nir_images

    return loaded_dates, img_arrays