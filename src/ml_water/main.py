from helpers.neural_network import predict_full_image
from helpers.initializations import create_model
from helpers.path import resolve_path
from helpers.constants import EVALUATION_IMAGE_PATH
from helpers.image_funcs import visualize_image, get_normalized_image


device = 'cpu'

model = create_model()

path = resolve_path(EVALUATION_IMAGE_PATH)

prediction_mask = predict_full_image(
  model=model,
  image_path=path,
  device=device,
)

image = get_normalized_image(path)

visualize_image(image, title='Full Image')

visualize_image(prediction_mask, title='Prediction of Full Image', cmap='gray')
