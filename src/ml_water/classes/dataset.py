import torch

from torch.utils.data import Dataset, DataLoader

import numpy as np

class Sentinel2Dataset(Dataset):
  def __init__(self, image_paths, mask_paths, transform=None):
    self.image_paths = image_paths
    self.mask_paths = mask_paths
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    # Load image and mask from the file paths
    image = np.load(self.image_paths[idx])
    mask = np.load(self.mask_paths[idx])

    # If there are transformations, apply them
    if self.transform:
      image = self.transform(image)

      # Convert image and mask to tensors
    image = torch.tensor(image).float() #.permute(2, 0, 1)  # Change channel order to (C, H, W)
    mask = torch.tensor(mask).unsqueeze(0).float()  # Add channel dimension for mask

    return image, mask
