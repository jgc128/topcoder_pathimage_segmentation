import numpy as np
import cv2
import torch
import torch.utils.data

from utils.io import load_image


class PathologicalImagesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(PathologicalImagesDataset, self).__init__()

        self.images_dir = data_dir.joinpath('images/')
        self.truth_dir = data_dir.joinpath('truth/')

        self.images = sorted(self.images_dir.iterdir())
        self.nb_images = len(self.images)

        if self.truth_dir.exists():
            self.masks = [self.truth_dir.joinpath(f'{img.stem}_mask.png') for img in self.images]
        else:
            self.masks = None

        self.transform = transform

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

        return image, mask

    def __len__(self):
        return self.nb_images