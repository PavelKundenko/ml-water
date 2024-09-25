from PIL import Image  # Use Pillow to save in formats like JPG, PNG
from typing import Any
import os
import shutil

import rasterio
import numpy as np

import matplotlib.pyplot as plt


def get_normalized_image(image_path: str) -> np.ndarray[Any, np.dtype]:
  # Load the RGB jp2 file
  with rasterio.open(image_path) as src:
    rgb_image = src.read([1, 2, 3])  # Read the three bands (R, G, B)

  # Transpose the array to (height, width, channels)
  rgb_image = np.transpose(rgb_image, (1, 2, 0))

  # Now you can use rgb_image for further processing, such as normalization
  rgb_image_normalized = rgb_image.astype(np.float32) / rgb_image.max()

  return rgb_image_normalized


def visualize_image(
    image_normalized: np.ndarray[Any, np.dtype],
    title: str = '',
    cmap=None
) -> None:
  plt.figure(figsize=(8, 8))
  plt.imshow(image_normalized, cmap=cmap)

  if title != '':
    plt.title(title)

  plt.axis('off')
  plt.show()


def save_image(image: np.ndarray, file_path: str, image_format="PNG"):
  # If the file exists, remove it
  if os.path.exists(file_path):
    os.remove(file_path)

  # Convert the image from 1-channel to 3-channel if necessary
  if len(image.shape) == 2:  # Grayscale image, we need to convert it to RGB
    image = np.stack([image] * 3, axis=-1)

  # Normalize the image to be between 0 and 255 for saving as a readable image
  image = (image * 255).astype(np.uint8)

  # Save the image using PIL
  Image.fromarray(image).save(file_path, format=image_format)


def split_image(image_normalized: np.ndarray[Any, np.dtype], patch_size=512):
  if len(image_normalized.shape) == 3:
    height, width, channels = image_normalized.shape
  elif len(image_normalized.shape) == 2:
    height, width = image_normalized.shape
    channels = None
  else:
    raise ValueError("Unsupported image shape: expected 2 or 3 dimensions, got {}".format(len(image_normalized.shape)))

  # Calculate the number of patches that fit in the height and width
  num_patches_x = width // patch_size
  num_patches_y = height // patch_size

  # Initialize a list to store the patches
  patches = []

  # Loop to extract patches
  for i in range(num_patches_y):
    for j in range(num_patches_x):
      # Calculate the start and end points for the patch
      start_y = i * patch_size
      start_x = j * patch_size
      end_y = start_y + patch_size
      end_x = start_x + patch_size

      # Extract the patch
      if channels:
        patch = image_normalized[start_y:end_y, start_x:end_x, :]
      else:
        patch = image_normalized[start_y:end_y, start_x:end_x]

      # Add the patch to the list
      patches.append(patch)

  # Convert the list of patches into a NumPy array if needed
  patches = np.array(patches)

  return patches


def reconstruct_image(patches, patch_size):
  # Determine the number of patches along one side (assuming square image)
  num_patches = int(np.sqrt(len(patches)))

  # Determine the number of channels from the first patch
  if len(patches[0].shape) == 3:
    channels = patches[0].shape[2]
  else:
    channels = None

  # Calculate the size of the full image
  image_size = num_patches * patch_size

  # Initialize the full image
  if channels:
    full_image = np.zeros((image_size, image_size, channels), dtype=np.float32)
  else:
    full_image = np.zeros((image_size, image_size), dtype=np.float32)

  # Fill the full image patch by patch
  index = 0
  for i in range(0, image_size, patch_size):
    for j in range(0, image_size, patch_size):
      if channels:
        full_image[i:i+patch_size, j:j+patch_size, :] = patches[index]
      else:
        full_image[i:i+patch_size, j:j+patch_size] = patches[index]
      index += 1

  return full_image


def save_patches(patches: list[np.ndarray[Any, np.dtype]], output_dir: str) -> None:
  # Check if the directory is empty
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Remove all files and folders in the directory

  # Recreate the directory
  os.makedirs(output_dir, exist_ok=True)

  # Loop through the patches and save each one as a .npy file
  for index, patch in enumerate(patches):
    patch_filename = f'{index}_patch.npy'
    patch_path = os.path.join(output_dir, patch_filename)
    np.save(patch_path, patch)  # Save the patch as a .npy file

    print(f"Saved {patch_filename} at {patch_path}")

  # Optional: Print the total number of patches saved
  print(f"Total patches saved: {len(patches)}")


def get_images_and_masks(image_path, mask_path, patches_number=8, start_index=0):
  image_paths = []  # Replace with actual paths
  mask_paths = []  # Replace with actual paths

  for i in range(start_index, start_index + patches_number - 1):
    image_paths.append(f'{image_path}/{i}_patch.npy')
    mask_paths.append(f'{mask_path}/{i}_patch.npy')

  return image_paths, mask_paths


def get_images_and_masks_random(
    image_path,
    mask_path,
    start_index,
    end_index,
    patches_number=8,
):
  image_paths = []  # Replace with actual paths
  mask_paths = []  # Replace with actual paths

  indices = np.random.randint(start_index, end_index, patches_number)

  print(f'Randomly selected image indexes: {indices}')

  for i in indices:
    image_paths.append(f'{image_path}/{i}_patch.npy')
    mask_paths.append(f'{mask_path}/{i}_patch.npy')

  return image_paths, mask_paths


def get_top_mask_patches(mask_folder, top_number=5):
  # List all mask patch files in the folder
  mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.npy')]

  # Calculate the number of "1" pixels in each patch
  ones_counts = []
  for mask_file in mask_files:
    mask = np.load(os.path.join(mask_folder, mask_file))
    ones_counts.append(np.sum(mask))

  # Identify the indices of the top_n patches with the most "1" pixels
  top_indices = np.argsort(ones_counts)[-top_number:][::-1]

  return top_indices.tolist()


def get_top_images_and_masks(
    image_path,
    mask_path,
    patches_number,
):
  # Replace with actual paths
  image_paths = []
  mask_paths = []

  top_patches = get_top_mask_patches(mask_folder=mask_path, top_number=patches_number)

  for i in top_patches:
    image_paths.append(f'{image_path}/{i}_patch.npy')
    mask_paths.append(f'{mask_path}/{i}_patch.npy')

  return image_paths, mask_paths
