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


class ImageMaskTransformsCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)

        return image, mask


class MaskToTensor():
    def __call__(self, mask):
        mask = torch.from_numpy(mask)
        return mask.float().div(255)
