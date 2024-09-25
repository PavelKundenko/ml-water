SELECTED_BAND_PACKAGE_ID = 2
SELECTED_MODEL_SNAPSHOT_ID = 5

TEST_DATA_BAND_PATH = f'test_data/band_{SELECTED_BAND_PACKAGE_ID}'  # Path to the test data directory

PATCH_SIZE = 512

RGB_BAND_PATH = f'{TEST_DATA_BAND_PATH}/bands/TCI_10m.jp2'  # TCI True Color band
B3_BAND_PATH = f'{TEST_DATA_BAND_PATH}/bands/B03_10m.jp2'  # B03 Green band
B8_BAND_PATH = f'{TEST_DATA_BAND_PATH}/bands/B08_10m.jp2'  # B08 NIR band
B11_BAND_PATH = f'{TEST_DATA_BAND_PATH}/bands/B11_20m.jp2'  # B11 SWIR band (20m resolution)

IMAGE_PATCHES_PATH = f'{TEST_DATA_BAND_PATH}/image_patches'  # Directory to save image patches
MASK_PATCHES_PATH = f'{TEST_DATA_BAND_PATH}/mask_patches'  # Directory to save mask patches
FULL_MASK_PATH = f'{TEST_DATA_BAND_PATH}/full_mask'  # Path to save the full mask

EVALUATION_DATA_PATH = 'evaluation_data'  # Path to the evaluation data directory

EVALUATION_IMAGE_PATH = f'{EVALUATION_DATA_PATH}/T36UXU_20240430T083601_TCI_10m.jp2'  # Path to the evaluation image

PREDICTION_IMAGE_PATH = f'{EVALUATION_DATA_PATH}/prediction_model_{SELECTED_MODEL_SNAPSHOT_ID}'

MODEL_PATH = f'models/model_{SELECTED_MODEL_SNAPSHOT_ID}.pth'  # Path to save the trained model
BEST_LOSS_PATH = 'models/best_loss.txt'  # Path to save the best loss value
