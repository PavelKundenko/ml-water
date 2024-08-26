import os
import torch

from ml_water.helpers.path import resolve_path
from ml_water.helpers.constants import MODEL_PATH


def load_state_nn(model):
  if os.path.exists(resolve_path(MODEL_PATH)):
    model.load_state_dict(torch.load(resolve_path(MODEL_PATH)))
    print(f"Model loaded successfully from {resolve_path(MODEL_PATH)}")


def save_state_nn(model):
  # Resolve the model path
  model_path = resolve_path(MODEL_PATH)

  # Ensure the directory exists
  directory = os.path.dirname(model_path)
  if not os.path.exists(directory):
    os.makedirs(directory)

  # Save the model state dictionary
  torch.save(model.state_dict(), model_path)
  print(f"Model saved successfully at {model_path}")
