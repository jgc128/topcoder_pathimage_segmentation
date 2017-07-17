from pathlib import Path

DATA_DIR = Path('./data')

DATASET_DIR = DATA_DIR.joinpath('raw/', 'PathImageSegmentation-data')
DATASET_TRAIN_DIR = DATASET_DIR.joinpath('training/')
DATASET_TEST_DIR = DATASET_DIR.joinpath('testing/')
