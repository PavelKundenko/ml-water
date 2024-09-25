from ml_water.helpers.initializations import create_model, create_dataloader
from ml_water.helpers.neural_network import evaluate_nn, train_nn_to_target_loss, train_nn_for_fixed_cycles
from ml_water.helpers.image_funcs import get_images_and_masks, get_top_images_and_masks
from ml_water.helpers.path import resolve_path
from ml_water.helpers.constants import IMAGE_PATCHES_PATH, MASK_PATCHES_PATH

device = 'cpu'

model = create_model()

train_images_indexes = (154, 438)

# train_image_paths, train_mask_paths = get_images_and_masks(
#   image_path=resolve_path(IMAGE_PATCHES_PATH),
#   mask_path=resolve_path(MASK_PATCHES_PATH),
#   start_index=train_images_indexes[0],
#   patches_number=train_images_indexes[1] - train_images_indexes[0] - 1,
# )

train_image_paths, train_mask_paths = get_top_images_and_masks(
  image_path=resolve_path(IMAGE_PATCHES_PATH),
  mask_path=resolve_path(MASK_PATCHES_PATH),
  patches_number=128,
)

train_nn_to_target_loss(
  model=model,
  target_loss=0.1,
  device=device,
  epochs=10,
  patch_size=16,
  image_paths=train_image_paths,
  mask_paths=train_mask_paths,
)

# train_nn_for_fixed_cycles(
#   model=model,
#   device=device,
#   epochs=10,
#   patch_size=16,
#   image_paths=train_image_paths,
#   mask_paths=train_mask_paths,
#   cycles=5,
# )

# train_image_paths, train_mask_paths = get_images_and_masks_random(
#   image_path=resolve_path(IMAGE_PATCHES_PATH),
#   mask_path=resolve_path(MASK_PATCHES_PATH),
#   start_index=train_images_indexes[0],
#   end_index=train_images_indexes[1],
#   patches_number=16,
# )
#
# train_dataloader = create_dataloader(train_image_paths, train_mask_paths)

# Training loop
# num_epochs = 20  # Example number of epochs

# train_nn(model, train_dataloader, num_epochs, device)

evaluation_images_indexes = (240, 255)

evaluation_image_paths, evaluation_mask_paths = get_images_and_masks(
  image_path=resolve_path(IMAGE_PATCHES_PATH),
  mask_path=resolve_path(MASK_PATCHES_PATH),
  start_index=evaluation_images_indexes[0],
  patches_number=8,
)

evaluation_dataloader = create_dataloader(evaluation_image_paths, evaluation_mask_paths)

evaluate_nn(model, evaluation_dataloader, device)
