from collections import defaultdict

import os
import logging

import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms

from sacred import Experiment

import config
from models.fcn import FCN32
from utils.torch.helpers import set_variable_repr, maybe_to_cuda
from utils.torch.datasets import PathologicalImagesDataset, PathologicalImagesDatasetMode
import utils.torch.transforms

ex = Experiment()


def create_data_loader(mode, batch_size=32, patch_size=0, augment=True, shuffle=True):
    transform = []

    if patch_size != 0:
        transform.append(utils.torch.transforms.SamplePatch(patch_size))

    if augment:
        transform.extend([
            utils.torch.transforms.RandomTranspose(),
            utils.torch.transforms.RandomVerticalFlip(),
            utils.torch.transforms.RandomHorizontalFlip(),
            utils.torch.transforms.Add(-50, 50, per_channel=False),
            utils.torch.transforms.ContrastNormalization(0.5, 1.5, per_channel=False),
            utils.torch.transforms.Rotate90n(),
            utils.torch.transforms.Rotate(-30, 30, mode='reflect'),
            utils.torch.transforms.CopyNumpy(),
        ])

    transform = utils.torch.transforms.ImageMaskTransformsCompose(transform)

    image_transform = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.70500564, 0.4902217, 0.6467339], std=[0.19247672, 0.20918619, 0.15601342]
        ),
    ]
    image_transform = torchvision.transforms.Compose(image_transform)

    mask_transform = [
        utils.torch.transforms.MaskToTensor()
    ]
    mask_transform = torchvision.transforms.Compose(mask_transform)

    data_set = PathologicalImagesDataset(
        config.DATASET_TRAIN_DIR, mode=mode,
        transform=transform, image_transform=image_transform, mask_transform=mask_transform
    )

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=1,
                                              pin_memory=torch.cuda.is_available())

    return data_loader


def train_model(model, data_loader_train, data_loader_val,
                learning_rate, nb_epochs, batch_size, regularization, checkpoint_filename):
    data_loaders = {
        'train': data_loader_train,
        'val': data_loader_val,
    }

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=regularization)

    model = maybe_to_cuda(model)

    j = 1
    loss_best = np.inf
    iteration = 0

    for epoch in range(nb_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss_bce = 0.0
            for j, (images, masks) in enumerate(data_loaders[phase], 1):
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

            model_saved_str = ''
            if phase == 'val' and epoch_loss_bce < loss_best:
                torch.save(model.state_dict(), checkpoint_filename)
                loss_best = epoch_loss_bce

                model_saved_str = '[model saved]'

            logging.info(f'Epoch {epoch} {phase}, loss: {epoch_loss_bce:.5f} {model_saved_str}')


@ex.config
def cfg():
    patch_size = 224

    regularization = 0.00001

    learning_rate = 0.001
    batch_size = 40
    nb_epochs = 50


@ex.main
def main(patch_size, regularization, learning_rate, batch_size, nb_epochs):
    set_variable_repr()

    model = FCN32(nb_classes=1)

    data_loader_train = create_data_loader(PathologicalImagesDatasetMode.Train, batch_size=batch_size,
                                           patch_size=patch_size, augment=True, shuffle=True)
    data_loader_val = create_data_loader(PathologicalImagesDatasetMode.Val, batch_size=batch_size,
                                         patch_size=patch_size, augment=True, shuffle=True)

    checkpoint_filename = str(config.MODELS_DIR.joinpath(f'{type(model).__name__}_{patch_size}.ckpt'))
    train_model(model, data_loader_train, data_loader_val, learning_rate, nb_epochs, batch_size, regularization,
                checkpoint_filename)


if __name__ == '__main__':
    ex.run_commandline()
