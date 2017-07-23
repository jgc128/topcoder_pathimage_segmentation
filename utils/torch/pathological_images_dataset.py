import logging
from enum import Enum

import numpy as np
import cv2
import torch
import torch.utils.data

from sklearn.model_selection import train_test_split

from utils.io import load_image


class PathologicalImagesDatasetMode(Enum):
    Train = 1
    Val = 2
    All = 3


class PathologicalImagesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode=PathologicalImagesDatasetMode.Train,
                 transform=None, image_transform=None, mask_transform=None):
        super(PathologicalImagesDataset, self).__init__()

        self.mode = mode

        self.images_dir = data_dir.joinpath('images/')
        self.truth_dir = data_dir.joinpath('truth/')

        images = sorted(self.images_dir.iterdir())

        if self.mode == PathologicalImagesDatasetMode.All:
            self.images = images
        else:
            images_train, images_val = train_test_split(images, test_size=0.2, random_state=42)
            if self.mode == PathologicalImagesDatasetMode.Train:
                self.images = images_train
            else:
                self.images = images_val

        self.nb_images = len(self.images)

        if self.truth_dir.exists():
            self.masks = [self.truth_dir.joinpath(f'{img.stem}_mask.png') for img in self.images]
        else:
            self.masks = None

        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        logging.info(f'Data: {self.mode} - {len(self.images)} images')

    def __getitem__(self, index):
        image_filename = self.images[index]
        image = load_image(image_filename)

        if self.masks is not None:
            mask_filename = self.masks[index]
            mask = load_image(mask_filename, grayscale=True)
        else:
            mask = 0

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return self.nb_images
