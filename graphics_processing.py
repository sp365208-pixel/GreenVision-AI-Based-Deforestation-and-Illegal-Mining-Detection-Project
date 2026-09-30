import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageColor
 
 
def bw_to_color(
    bw_image: np.ndarray, black_substitute: str, white_substitute: str
    ) -> np.ndarray:
    black_substitute_rgb = ImageColor.getcolor(black_substitute, "RGB")
    white_substitute_rgb = ImageColor.getcolor(white_substitute, "RGB")
    height, width = bw_image.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    # Fix: handle float (0-1) and uint8 (0-255) both
    color_image[bw_image == 0] = black_substitute_rgb
    color_image[bw_image > 0] = white_substitute_rgb
    return color_image
 
 
def smooth_and_vectorize(array, hex_code, smoothing=9, opacity=0.5):
    # Fix: convert float mask (0-1) to uint8 (0-255)
    if array.dtype != np.uint8:
        array = (array * 255).astype(np.uint8)
 
    array_resized = cv2.resize(array, (512, 512))
    array_rgba = np.zeros((512, 512, 4), dtype=np.uint8)
    array_rgba[:, :, 0] = array_resized
    array_rgba[:, :, 1] = array_resized
    array_rgba[:, :, 2] = array_resized
    # Fix: forest=1(white=255) should be visible, non-forest=0(black) transparent
    array_rgba[:, :, 3] = np.where(array_resized > 127, 255, 0)
 
    image = Image.fromarray(array_rgba)
    med_img = image.filter(ImageFilter.MedianFilter(size=smoothing))
 
    smoothed_image = med_img.convert("L")
    smoothed_array = np.array(smoothed_image)
 
    _, binary_image = cv2.threshold(smoothed_array, 127, 255, cv2.THRESH_BINARY)
    mask = Image.fromarray(binary_image).convert("L")
 
    hex_code = hex_code.lstrip('#')
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    opacity_val = int(opacity * 255)
    rgba = (r, g, b, opacity_val)
    color_img = Image.new("RGBA", image.size, rgba)
 
    smooth_coloured_vector = Image.composite(
        color_img,
        Image.new("RGBA", image.size, (0, 0, 0, 0)),
        mask
    )
    return smooth_coloured_vector.convert("RGBA")
 
 
def overlay_vector_on_image(
    vector: np.ndarray, image, vector_hex_code: str
    ) -> Image.Image:
    """Paste the vector on the image with the given hex code."""
    smoothed_vector = smooth_and_vectorize(
        array=vector,
        hex_code=vector_hex_code
    )
    # Handle numpy array or PIL image
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, :3]
        converted_image = Image.fromarray(image, mode='RGB').convert('RGBA')
    elif isinstance(image, Image.Image):
        converted_image = image.convert('RGBA')
    else:
        converted_image = Image.fromarray(np.array(image)).convert('RGBA')
 
    overlayed_image = Image.alpha_composite(
        im1=converted_image,
        im2=smoothed_vector
    )
    return overlayed_image