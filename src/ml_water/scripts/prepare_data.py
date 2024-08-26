import numpy as np

from ml_water.helpers.image_funcs import split_image, get_normalized_image, visualize_image, save_patches
from ml_water.helpers.path import resolve_path
from ml_water.helpers.constants import RGB_BAND_PATH, IMAGE_PATCHES_PATH

rgb_normalized_image = get_normalized_image(resolve_path(RGB_BAND_PATH))

patches = split_image(rgb_normalized_image, patch_size=512)

output_dir = resolve_path(IMAGE_PATCHES_PATH)  # Replace with your desired directory

save_patches(patches, output_dir)

visualize_image(np.load(f'{output_dir}/1_patch.npy'), title='Sample Patch 1')
