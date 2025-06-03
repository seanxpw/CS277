import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Input and output paths
input_dir = Path("/home/CS277/imagenet_images")
output_dir = Path("dataset")
output_dir.mkdir(exist_ok=True)

sobel_ksize = 3  # Kernel size for Sobel operator

# Loop through all image files
for image_path in tqdm(list(input_dir.glob("*"))):
    if not image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        continue

    # Read image
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Apply Sobel operator
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)

    # Normalize to 0–255 and save
    sobel_norm = np.zeros_like(sobel_mag, dtype=np.uint8)
    cv2.normalize(sobel_mag, sobel_norm, 0, 255, cv2.NORM_MINMAX)

    # Save original image (copy)
    img.save(output_dir / image_path.name)

    # Save Sobel edge map
    sobel_filename = image_path.stem + "_sobel.png"
    cv2.imwrite(str(output_dir / sobel_filename), sobel_norm)
