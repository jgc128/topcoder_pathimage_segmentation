import abc

import numpy as np


class BaseImageMaskTransformer(object):
    def __init__(self):
        super(BaseImageMaskTransformer, self).__init__()

    @abc.abstractmethod
    def transform(self, image):
        """Transform the provided object"""

    def __call__(self, image, mask):
        if np.random.rand() >= 0.5:
            image = self.transform(image)

            if isinstance(mask, np.ndarray):
                mask = self.transform(mask)

        return image, mask


class RandomVerticalFlip(BaseImageMaskTransformer):
    def __init__(self):
        super(RandomVerticalFlip, self).__init__()

    def transform(self, image):
        image = np.flipud(image)
        return image


class RandomHorizontalFlip(BaseImageMaskTransformer):
    def __init__(self):
        super(RandomHorizontalFlip, self).__init__()

    def transform(self, image):
        image = np.fliplr(image)
        return image


class RandomTranspose(BaseImageMaskTransformer):
    def __init__(self):
        super(RandomTranspose, self).__init__()

    def transform(self, image):
        transpose_axis = [1, 0]
        if len(image.shape) == 3:
            transpose_axis.append(2)
            
        image = np.transpose(image, transpose_axis)
        return image


class ImageMaskTransformsCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)

        return image, mask
