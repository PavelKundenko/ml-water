from datetime import datetime

from helpers.neural_network import predict_full_image
from helpers.initializations import create_model
from helpers.path import resolve_path
from helpers.constants import EVALUATION_IMAGE_PATH, PREDICTION_IMAGE_PATH
from helpers.image_funcs import visualize_image, get_normalized_image, save_image


device = 'cpu'

model = create_model()

evaluation_image_path = resolve_path(EVALUATION_IMAGE_PATH)

prediction_mask = predict_full_image(
  model=model,
  image_path=evaluation_image_path,
  device=device,
)

prediction_mask_dir = resolve_path(PREDICTION_IMAGE_PATH, create_dir=True)

save_image(prediction_mask, f'{prediction_mask_dir}/{datetime.now().isoformat()}.png')

visualize_image(get_normalized_image(evaluation_image_path), title='Full Image')

visualize_image(prediction_mask, title='Prediction of Full Image', cmap='gray')
