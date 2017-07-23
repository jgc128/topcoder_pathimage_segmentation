import numpy as np

import config

from utils.torch.datasets import PathologicalImagesDataset


def test_len():
    dataset = PathologicalImagesDataset(config.DATASET_TRAIN_DIR)

    assert len(dataset) == 168


def test_test_load_image_and_mask_for_train():
    dataset = PathologicalImagesDataset(config.DATASET_TRAIN_DIR)
    image, mask = dataset[0]

    assert image.dtype == np.uint8
    assert image.shape == (500, 500, 3)

    assert mask.dtype == np.uint8
    assert mask.shape == (500, 500)


def test_load_image_only_for_test():
    dataset = PathologicalImagesDataset(config.DATASET_TEST_DIR)
    image, mask = dataset[0]

    assert image.dtype == np.uint8
    assert image.shape == (500, 500, 3)

    assert isinstance(mask, int)
    assert mask == 0
