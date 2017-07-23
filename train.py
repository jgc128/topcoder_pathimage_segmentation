import os
import logging

import numpy as np

from tqdm import tqdm

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms

import config
from models.fcn import FCN32
from utils.torch.pathological_images_dataset import PathologicalImagesDataset
from utils.torch.transforms import MaskToTensor, ImageMaskTransformsCompose, SamplePatch, RandomTranspose, \
    RandomVerticalFlip, RandomHorizontalFlip, CopyNumpy


def maybe_to_cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()

    return obj


def main():
    patch_size = 224
    transform = [
        SamplePatch(patch_size),
        RandomTranspose(),
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        CopyNumpy(),
    ]
    transform = ImageMaskTransformsCompose(transform)

    image_transform = [
        torchvision.transforms.ToTensor(),
    ]
    image_transform = torchvision.transforms.Compose(image_transform)

    mask_transform = [
        MaskToTensor()
    ]
    mask_transform = torchvision.transforms.Compose(mask_transform)

    data_set = PathologicalImagesDataset(
        config.DATASET_TRAIN_DIR,
        transform=transform, image_transform=image_transform, mask_transform=mask_transform
    )

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=True, num_workers=1,
                                              pin_memory=torch.cuda.is_available())

    model = FCN32(nb_classes=1)

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.00001)

    model = maybe_to_cuda(model)

    j = 1
    loss_best = np.inf
    iteration = 0
    for epoch in range(10):
        for phase in ['train', ]:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss_bce = 0.0
            for j, (images, masks) in tqdm(enumerate(data_loader, 1), total=len(data_loader),
                                           desc=f'Epoch {epoch} {phase}'):
                images = torch.autograd.Variable(maybe_to_cuda(images))
                masks = torch.autograd.Variable(maybe_to_cuda(masks))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(images)
                outputs = outputs.squeeze()

                loss_bce = loss_fn(outputs, masks)

                if phase == 'train':
                    loss_bce.backward()
                    optimizer.step()

                running_loss_bce += loss_bce.data[0]

                del loss_bce
                del outputs

            epoch_loss_bce = running_loss_bce / j

            print(f'Epoch {epoch} {phase}, loss: {epoch_loss_bce}')


if __name__ == '__main__':
    Variable.__repr__ = lambda x: f'Variable {tuple(x.size())}'
    main()
