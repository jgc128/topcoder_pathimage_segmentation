import abc
from enum import Enum

import numpy as np
import cv2

import torch


class ImageMaskTransformMode(Enum):
    Image = 1
    Mask = 2


class BaseImageMaskTransformer(object):
    def __init__(self):
        super(BaseImageMaskTransformer, self).__init__()

        self.apply_always = False

    @abc.abstractmethod
    def transform(self, image, mode):
        """Transform the provided object"""

    def __call__(self, image, mask):
        if np.random.rand() >= 0.5 or self.apply_always:
            image = self.transform(image, mode=ImageMaskTransformMode.Image)

            if isinstance(mask, np.ndarray):
                mask = self.transform(mask, mode=ImageMaskTransformMode.Mask)

        return image, mask


class RandomVerticalFlip(BaseImageMaskTransformer):
    def transform(self, image, mode):
        image = np.flipud(image)
        return image


class RandomHorizontalFlip(BaseImageMaskTransformer):
    def transform(self, image, mode):
        image = np.fliplr(image)
        return image


class RandomTranspose(BaseImageMaskTransformer):
    def transform(self, image, mode):
        transpose_axis = [1, 0]
        if len(image.shape) == 3:
            transpose_axis.append(2)

        image = np.transpose(image, transpose_axis)
        return image


class Resize(BaseImageMaskTransformer):
    def __init__(self, size):
        super(Resize, self).__init__()

        self.size = size

        self.apply_always = True

    def transform(self, image, mode):
        if mode == ImageMaskTransformMode.Image:
            interpolation = cv2.INTER_LINEAR
        else:
            interpolation = cv2.INTER_NEAREST

        image = cv2.resize(image, (self.size, self.size), interpolation=interpolation)

        return image


class CopyNumpy(BaseImageMaskTransformer):
    def __init__(self):
        super(CopyNumpy, self).__init__()

        self.apply_always = True

    def transform(self, image, mode):
        return np.copy(image)


class SamplePatch(BaseImageMaskTransformer):
    def __init__(self, patch_size):
        super(SamplePatch, self).__init__()

        self.patch_size = patch_size

        self.apply_always = True

        self._patch_coordinate_h = 0
        self._patch_coordinate_w = 0

    def transform(self, image, mode):
        # sample coordinates to apply to both image and mask
        if mode == ImageMaskTransformMode.Image:
            image_height = image.shape[0]
            image_width = image.shape[1]

            max_height = image_height - self.patch_size
            max_width = image_width - self.patch_size

            self._patch_coordinate_h = np.random.randint(0, max_height)
            self._patch_coordinate_w = np.random.randint(0, max_width)

        c_h = (self._patch_coordinate_h, self._patch_coordinate_h + self.patch_size)
        c_w = (self._patch_coordinate_w, self._patch_coordinate_w + self.patch_size)
        image = image[c_h[0]:c_h[1], c_w[0]:c_w[1]]

        return image


class ImageMaskTransformsCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for transform_step in self.transforms:
            image, mask = transform_step(image, mask)

        return image, mask


class MaskToTensor():
    def __call__(self, mask):
        mask = torch.from_numpy(mask)
        return mask.float().div(255)
