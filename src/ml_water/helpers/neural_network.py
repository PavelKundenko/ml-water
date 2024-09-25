import random
import time
import torch
import asyncio
import torch.nn as nn
import numpy as np
from sklearn.metrics import jaccard_score, precision_score, recall_score

from ml_water.helpers.initializations import create_dataloader
from ml_water.helpers.nn_state import save_state_nn
from ml_water.helpers.image_funcs import split_image, reconstruct_image, get_normalized_image
from ml_water.helpers.constants import PATCH_SIZE


def train_nn(model, dataloader, num_epochs, device):
  # Define the loss function and optimizer
  criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary segmentation
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer with learning rate

  # Load the previous best loss
  last_loss = None

  # Measure the total training time
  total_start_time = time.time()

  for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Start time of the epoch

    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, masks in dataloader:
      # Move images and masks to the same device as model
      images, masks = images.to(device), masks.to(device)

      # Forward pass: compute predictions
      outputs = model(images)

      # Compute the loss
      loss = criterion(outputs, masks)

      # Backward pass: compute gradients
      optimizer.zero_grad()
      loss.backward()

      # Update the weights
      optimizer.step()

      # Accumulate the loss
      running_loss += loss.item() * images.size(0)

    # Compute average loss over the epoch
    epoch_loss = running_loss / len(dataloader.dataset)


    last_loss = epoch_loss

    epoch_end_time = time.time()  # End time of the epoch
    epoch_duration = epoch_end_time - epoch_start_time

    if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
      save_state_nn(model)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Duration: {epoch_duration:.2f} seconds")

  total_end_time = time.time()  # End time of the training
  total_duration = total_end_time - total_start_time

  print(f"Training completed in {total_duration:.2f} seconds")

  return last_loss


def train_nn_to_target_loss(
    model,
    target_loss,
    device,
    epochs,
    patch_size,
    image_paths,
    mask_paths,
):
  current_loss = float('inf')
  start_time = time.time()  # Start time tracking

  iteration = 1

  while current_loss > target_loss:
    print(f'Starting training iteration #{iteration}')

    # Randomly select patch_size items from image_paths and mask_paths
    selected_indices = random.sample(range(len(image_paths)), patch_size)
    selected_image_paths = [image_paths[i] for i in selected_indices]
    selected_mask_paths = [mask_paths[i] for i in selected_indices]

    print(f'Selected images: {selected_indices}')

    # Create the dataloader with the selected items
    dataloader = create_dataloader(selected_image_paths, selected_mask_paths)

    current_loss = train_nn(model, dataloader, epochs, device)

    print(f'Training iteration #{iteration} completed')
    iteration += 1

  end_time = time.time()  # End time tracking
  duration = end_time - start_time  # Calculate duration

  print(f"Training completed. Target loss achieved: {current_loss:.4f}. Duration: {duration:.2f} seconds")


def train_nn_for_fixed_cycles(
    model,
    device,
    epochs,
    patch_size,
    image_paths,
    mask_paths,
    cycles
):
  start_time = time.time()  # Start time tracking

  for iteration in range(1, cycles + 1):
    print(f'Starting training iteration #{iteration}')

    # Randomly select patch_size items from image_paths and mask_paths
    selected_indices = random.sample(range(len(image_paths)), patch_size)
    selected_image_paths = [image_paths[i] for i in selected_indices]
    selected_mask_paths = [mask_paths[i] for i in selected_indices]

    print(f'Selected images for iteration #{iteration}: {selected_indices}')

    # Create the dataloader with the selected items
    dataloader = create_dataloader(selected_image_paths, selected_mask_paths)

    # Train the model for the specified number of epochs
    current_loss = train_nn(model, dataloader, epochs, device)

    print(f'Training iteration #{iteration} completed. Loss: {current_loss:.4f}')

  end_time = time.time()  # End time tracking
  duration = end_time - start_time  # Calculate duration

  print(f"Training completed after {cycles} cycles. Duration: {duration:.2f} seconds")


# Function to calculate metrics
def evaluate_nn(model, dataloader, device):
  model.eval()  # Set the model to evaluation mode
  all_preds = []
  all_labels = []

  with torch.no_grad():  # Disable gradient calculation
    for images, masks in dataloader:
      images, masks = images.to(device), masks.to(device)
      outputs = model(images)

      preds = (outputs > 0.5).float()  # Apply threshold to get binary predictions
      all_preds.append(preds.cpu().numpy())
      all_labels.append(masks.cpu().numpy())

  all_preds = np.concatenate(all_preds).ravel()
  all_labels = np.concatenate(all_labels).ravel()

  # Calculate metrics
  iou = jaccard_score(all_labels, all_preds, average='binary')
  precision = precision_score(all_labels, all_preds, average='binary')
  recall = recall_score(all_labels, all_preds, average='binary')

  print(f"IoU: {iou:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")


def predict(model, device, image_path=None, image_data=None):
  # model.eval()  # Set the model to evaluation mode

  image = None

  # Load and preprocess the image
  if image_path is not None:
    image = np.load(image_path)
    image = image.astype(np.float32) / image.max()
  elif image_data is not None:
    image = image_data

  image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

  # Move to the same device as model
  image = image.to(device)

  with torch.no_grad():
    output = model(image)
    pred = (output > 0.5).float()  # Apply threshold to get binary prediction

  return pred.squeeze().cpu().numpy()  # Convert to numpy array


def predict_full_image(model, image_path, device, patch_size=PATCH_SIZE):
  # Load and normalize the full image using the existing function
  image = get_normalized_image(image_path)

  # Split the image into patches using the existing function
  patches = split_image(image, patch_size)

  # Predict each patch and store results
  predicted_patches = []

  for patch in patches:
    if len(predicted_patches) % 10 == 0:
      print(f'Processing patch {len(predicted_patches) + 1} / {len(patches)}. Done {len(predicted_patches) / len(patches) * 100:.2f}%')

    pred = predict(model=model, device=device, image_data=patch)

    predicted_patches.append(pred)

  # Reconstruct the full predicted mask from patches
  full_predicted_mask = reconstruct_image(predicted_patches, patch_size)

  return full_predicted_mask


async def predict_patch_async(model, patch, device, index):
  loop = asyncio.get_event_loop()
  pred = await loop.run_in_executor(None, predict, model, device, None, patch)
  return {index: pred}


async def predict_full_image_async(model, image_path, device, patch_size=PATCH_SIZE):
  # Load and normalize the full image using the existing function
  image = get_normalized_image(image_path)

  # Split the image into patches using the existing function
  patches = split_image(image, patch_size)

  # Asynchronously predict each patch
  tasks = []
  for idx, patch in enumerate(patches):
    if idx % 10 == 0:
      print(f'Processing patch {idx + 1} / {len(patches)}. Done {idx / len(patches) * 100:.2f}%')

    task = predict_patch_async(model, patch, device, idx)
    tasks.append(task)

  # Gather all predictions
  predicted_dicts = await asyncio.gather(*tasks)

  # Combine the results into a sorted list based on the index
  predicted_patches = [predicted_dict[idx] for predicted_dict in sorted(predicted_dicts, key=lambda x: list(x.keys())[0]) for idx in predicted_dict]

  # Reconstruct the full predicted mask from patches
  full_predicted_mask = reconstruct_image(predicted_patches, patch_size)

  return full_predicted_mask
