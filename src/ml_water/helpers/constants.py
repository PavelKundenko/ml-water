SELECTED_BAND_PACKAGE_ID = 2
SELECTED_MODEL_SNAPSHOT_ID = 3

TEST_DATA_PATH = 'test_data'  # Path to the test data directory

PATCH_SIZE = 512

RGB_BAND_PATH = f'{TEST_DATA_PATH}/band_{SELECTED_BAND_PACKAGE_ID}/bands/TCI_10m.jp2'  # TCI True Color band
B3_BAND_PATH = f'{TEST_DATA_PATH}/band_{SELECTED_BAND_PACKAGE_ID}/bands/B03_10m.jp2'  # B03 Green band
B8_BAND_PATH = f'{TEST_DATA_PATH}/band_{SELECTED_BAND_PACKAGE_ID}/bands/B08_10m.jp2'  # B08 NIR band
B11_BAND_PATH = f'{TEST_DATA_PATH}/band_{SELECTED_BAND_PACKAGE_ID}/bands/B11_20m.jp2'  # B11 SWIR band (20m resolution)

IMAGE_PATCHES_PATH = f'{TEST_DATA_PATH}/band_{SELECTED_BAND_PACKAGE_ID}/image_patches'  # Directory to save image patches
MASK_PATCHES_PATH = f'{TEST_DATA_PATH}/band_{SELECTED_BAND_PACKAGE_ID}/mask_patches'  # Directory to save mask patches

EVALUATION_DATA_PATH = 'evaluation_data'  # Path to the evaluation data directory

EVALUATION_IMAGE_PATH = f'{EVALUATION_DATA_PATH}/T36UXU_20240430T083601_TCI_10m.jp2'  # Path to the evaluation image

MODEL_PATH = f'models/model_{SELECTED_MODEL_SNAPSHOT_ID}.pth'  # Path to save the trained model
BEST_LOSS_PATH = 'models/best_loss.txt'  # Path to save the best loss value
