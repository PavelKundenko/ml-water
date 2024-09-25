from torchsummary import summary
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from ml_water.classes.unet import UNet
from ml_water.classes.dataset import Sentinel2Dataset

from ml_water.helpers.nn_state import load_state_nn
from ml_water.helpers.constants import PATCH_SIZE


def create_model(in_channels=3, out_channels=1):
  model = UNet(in_channels=in_channels, out_channels=out_channels)

  load_state_nn(model)

  summary(model, input_size=(in_channels, PATCH_SIZE, PATCH_SIZE))

  return model


def create_dataloader(image_paths, mask_paths):
  # Define transformations (if needed)
  transform = transforms.Compose([
    transforms.ToTensor()
  ])

  # Create the dataset
  dataset = Sentinel2Dataset(image_paths=image_paths, mask_paths=mask_paths, transform=transform)

  batch_size = 8  # Example batch size
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

  return dataloader
