import rasterio

import numpy as np
import cv2

from ml_water.helpers.path import resolve_path
from ml_water.helpers.image_funcs import visualize_image, split_image, save_patches, get_normalized_image
from ml_water.helpers.constants import B3_BAND_PATH, MASK_PATCHES_PATH, RGB_BAND_PATH, B8_BAND_PATH

# with rasterio.open(resolve_path(B3_BAND_PATH)) as src:
#   b3_band = src.read(1).astype(np.float32)  # Green band
#
# with rasterio.open(resolve_path(B11_BAND_PATH)) as src:
#   b11_band = src.read(1).astype(np.float32)  # SWIR band
#
# # If B11 is at 20m resolution, it needs to be resampled to 10m to match B03
# # Resampling B11 to match B03 resolution (if needed)
# if b11_band.shape != b3_band.shape:
#   b11_band = cv2.resize(b11_band, (b3_band.shape[1], b3_band.shape[0]), interpolation=cv2.INTER_CUBIC)
#
# # Apply the formula (B3 - B11) / (B3 + B11)
# ndi_image = (b3_band - b11_band) / (b3_band + b11_band)
#
# # Handle division by zero (optional: set it to 0 or another value)
# ndi_image = np.nan_to_num(ndi_image)
#
# threshold = 0  # You can adjust this threshold based on your specific case
#
# # Create a binary mask: water pixels are 1, others are 0
# water_mask = np.where(ndi_image > threshold, 1, 0).astype(np.uint8)
#
# patches = split_image(water_mask, patch_size=512)
#
# output_dir = resolve_path(MASK_PATCHES_PATH)
#
# save_patches(patches, output_dir)

# visualize_image(np.load(f'{output_dir}/300_patch.npy'), title='Sample Mask Patch 1', cmap='gray')
# visualize_image(np.load(f'{resolve_path(IMAGE_PATCHES_PATH)}/300_patch.npy'), title='Sample Mask Patch 1')

# Load the Green band (B3)
with rasterio.open(resolve_path(B3_BAND_PATH)) as src:
  b3_band = src.read(1).astype(np.float32)  # Green band

# Load the NIR band (B8)
with rasterio.open(resolve_path(B8_BAND_PATH)) as src:
  b8_band = src.read(1).astype(np.float32)  # Near-Infrared band

# If B8 is at 20m resolution, it needs to be resampled to 10m to match B3
# Resampling B8 to match B3 resolution (if needed)
if b8_band.shape != b3_band.shape:
  b8_band = cv2.resize(b8_band, (b3_band.shape[1], b3_band.shape[0]), interpolation=cv2.INTER_CUBIC)

# Apply the NDWI formula (B3 - B8) / (B3 + B8)
ndwi_image = (b3_band - b8_band) / (b3_band + b8_band)

# Handle division by zero (optional: set it to 0 or another value)
ndwi_image = np.nan_to_num(ndwi_image)

threshold = 0  # You can adjust this threshold based on your specific case

# Create a binary mask: water pixels are 1, others are 0
water_mask = np.where(ndwi_image > threshold, 1, 0).astype(np.uint8)

# Split the mask into patches
patches = split_image(water_mask, patch_size=512)

# Save the patches to the specified output directory
output_dir = resolve_path(MASK_PATCHES_PATH)
save_patches(patches, output_dir)

visualize_image(get_normalized_image(resolve_path(RGB_BAND_PATH)), title='RGB Image')

visualize_image(water_mask, title='Mask', cmap='gray')
