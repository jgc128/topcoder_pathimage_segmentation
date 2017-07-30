import logging
from enum import Enum

import numpy as np
import torch.utils.data

from sklearn.model_selection import train_test_split, KFold

from utils.io import load_image


class PathologicalImagesDatasetMode(Enum):
    Train = 1
    Val = 2
    All = 3


class PathologicalImagesDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode=PathologicalImagesDatasetMode.Train, fold_number=0, nb_folds=5,
                 transform=None, image_transform=None, mask_transform=None):
        super(PathologicalImagesDataset, self).__init__()

        self.mode = mode
        self.fold_number = fold_number

        self.images_dir = data_dir.joinpath('images/')
        self.truth_dir = data_dir.joinpath('truth/')

        images = np.array(sorted(self.images_dir.iterdir()))

        if self.mode == PathologicalImagesDatasetMode.All:
            self.images = images
        else:
            idx_all = np.arange(len(images))

            kf = KFold(n_splits=nb_folds, shuffle=True, random_state=42)
            folds = [(idx_train, idx_val) for idx_train, idx_val in kf.split(idx_all)]
            target_fold = folds[fold_number]

            idx_train, idx_val = target_fold[0], target_fold[1]

            images_train, images_val = images[idx_train], images[idx_val]
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

        logging.info(f'Data: {self.mode}, fold {self.fold_number} - {self.nb_images} images')

    def _load_image_and_mask(self, index):
        image_filename = self.images[index]
        image = load_image(image_filename)

        if self.masks is not None:
            mask_filename = self.masks[index]
            mask = load_image(mask_filename, grayscale=True)
        else:
            mask = 0

        return image, mask

    def _apply_transforms(self, image, mask):
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None and isinstance(mask, np.ndarray):
            mask = self.mask_transform(mask)

        return image, mask

    def __getitem__(self, index):
        image, mask = self._load_image_and_mask(index)
        image, mask = self._apply_transforms(image, mask)

        return image, mask

    def __len__(self):
        return self.nb_images


class DeterministicPatchesDataset(PathologicalImagesDataset):
    def __init__(self, patch_size, *args, **kwargs):
        super(DeterministicPatchesDataset, self).__init__(*args, **kwargs)

        self.patch_size = patch_size

        # determine the size of the image
        # image0 = load_image(self.images[0])
        # self.image_height = image0.shape[0]
        # self.image_width = image0.shape[1]
        self.image_height = 500
        self.image_width = 500

        # generate patches for all images
        if self.patch_size != 0:
            patches_coordinates = self._get_patches_coordinates(self.image_height, self.image_width, self.patch_size,
                                                                overlap=0.5)
        else:
            patches_coordinates = [(0, 0)]

        self.patches = [
            (image_idx, patch)
            for image_idx in range(self.nb_images)
            for patch in patches_coordinates
        ]

        self.nb_patches = len(self.patches)
        logging.info(f'Data: {self.mode} - {self.nb_patches} patches')

    def _get_patches_coordinates(self, image_height, image_width, patch_size, overlap=0.5):
        step_size = (int(patch_size * overlap), int(patch_size * overlap))
        nb_steps_height = (image_height - patch_size) // step_size[0] + 1
        nb_steps_width = (image_width - patch_size) // step_size[1] + 1

        patches_coordinates = []

        # tile
        for i in range(nb_steps_height):
            for j in range(nb_steps_width):
                patch_coord = (i * step_size[0], j * step_size[1])
                patches_coordinates.append(patch_coord)

        # leftovers - width
        for i in range(nb_steps_height):
            patch_coord = (i * step_size[0], image_width - patch_size)
            patches_coordinates.append(patch_coord)

        # leftovers - height
        for j in range(nb_steps_width):
            patch_coord = (image_height - patch_size, j * step_size[1])
            patches_coordinates.append(patch_coord)

        # leftovers - bottom right corner
        patch_coord = (image_height - patch_size, image_width - patch_size)
        patches_coordinates.append(patch_coord)

        return patches_coordinates

    def __getitem__(self, index):
        patch_info = self.patches[index]
        image_idx = patch_info[0]

        image, mask = self._load_image_and_mask(image_idx)

        if self.patch_size != 0:
            # get patch
            c_h = (patch_info[1][0], patch_info[1][0] + self.patch_size)
            c_w = (patch_info[1][1], patch_info[1][1] + self.patch_size)

            image = image[c_h[0]:c_h[1], c_w[0]:c_w[1]]
            if isinstance(mask, np.ndarray):
                mask = mask[c_h[0]:c_h[1], c_w[0]:c_w[1]]

        image, mask = self._apply_transforms(image, mask)

        return image, mask

    def __len__(self):
        return self.nb_patches


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, images, masks=None, transform=None, image_transform=None, mask_transform=None):
        super(TransformDataset, self).__init__()

        self.images = images
        self.masks = masks

        self.nb_images = len(self.images)

        self.transform = transform
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        logging.info(f'Dataset: {self.nb_images} images')

    def _apply_transforms(self, image, mask):
        if self.transform is not None:
            image, mask = self.transform(image, mask)

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None and isinstance(mask, np.ndarray):
            mask = self.mask_transform(mask)

        return image, mask

    def __getitem__(self, index):
        image = self.images[index]
        if self.masks is not None:
            mask = self.masks[index]
        else:
            mask = 0

        image, mask = self._apply_transforms(image, mask)

        return image, mask

    def __len__(self):
        return self.nb_images
