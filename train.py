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
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

import config
from models import FCN32, UNet
from utils.torch.helpers import set_variable_repr, maybe_to_cuda
from utils.torch.datasets import PathologicalImagesDataset, PathologicalImagesDatasetMode
import utils.torch.transforms
from utils.torch.layers import CenterCrop2d
from utils.torch.losses import DiceWithLogitsLoss
from utils.torch.transforms import MakeBorder

ex = Experiment()


def create_data_loader(mode, batch_size=32, patch_size=0, make_border=0, augment=True, shuffle=True):
    transform = []

    if patch_size != 0:
        transform.append(utils.torch.transforms.SamplePatch(patch_size))

    if make_border != 0:
        transform.append(MakeBorder(border_size=make_border))

    if augment:
        transform.extend([
            utils.torch.transforms.RandomTranspose(),
            utils.torch.transforms.RandomVerticalFlip(),
            utils.torch.transforms.RandomHorizontalFlip(),
            # utils.torch.transforms.Add(-50, 50, per_channel=False),
            # utils.torch.transforms.ContrastNormalization(0.5, 1.5, per_channel=False),
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
                learning_rate, nb_epochs, batch_size, make_border, use_dice, regularization, checkpoint_filename):
    data_loaders = {
        'train': data_loader_train,
        'val': data_loader_val,
    }

    loss_fn_bce = torch.nn.BCEWithLogitsLoss()
    loss_fn_dice = DiceWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=regularization)
    lr_scheduler = MultiStepLR(optimizer, milestones=[200, 500, 700], gamma=0.1)

    model = maybe_to_cuda(model)

    j = 1
    loss_best = np.inf
    iteration = 0

    crop_outputs = None
    if make_border != 0:
        crop_outputs = CenterCrop2d(make_border)

    for epoch in range(nb_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            if phase == 'train':
                lr_scheduler.step()

            # TODO: optimize losses
            running_loss_bce = 0.0
            running_loss_dice = 0.0
            running_loss_total = 0.0
            for j, (images, masks) in enumerate(data_loaders[phase], 1):
                images = torch.autograd.Variable(maybe_to_cuda(images))
                masks = torch.autograd.Variable(maybe_to_cuda(masks))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(images)
                outputs = outputs.squeeze()

                if crop_outputs is not None:
                    outputs = crop_outputs(outputs)

                loss_bce = loss_fn_bce(outputs, masks)
                loss_dice = loss_fn_dice(outputs, masks)
                if use_dice:
                    loss_total = loss_bce + loss_dice
                else:
                    loss_total = loss_bce

                if phase == 'train':
                    loss_total.backward()
                    optimizer.step()

                running_loss_bce += loss_bce.data[0]
                running_loss_dice += loss_dice.data[0]
                running_loss_total += loss_total.data[0]

                del loss_bce
                del loss_dice
                del loss_total
                del outputs

            epoch_loss_bce = running_loss_bce / j
            epoch_loss_dice = running_loss_dice / j
            epoch_loss_total = running_loss_total / j

            log_str = f'Epoch {epoch} {phase}, ' \
                      f'bce: {epoch_loss_bce:.3f}, dice: {epoch_loss_dice:.3f}, total: {epoch_loss_total:.3f}'

            if phase == 'val' and epoch_loss_total < loss_best:
                torch.save(model.state_dict(), checkpoint_filename)
                loss_best = epoch_loss_total

                log_str += ' [model saved]'

            logging.info(log_str)
            append_log_file('training.log',
                            f'{epoch}\t{phase}\t{epoch_loss_bce:.3f}\t{epoch_loss_dice:.3f}\t{epoch_loss_total:.3f}')


def create_model(model_name, model_params):
    if model_name == 'fcn':
        model_class = FCN32
    elif model_name == 'unet':
        model_class = UNet
    else:
        raise ValueError(f'Unknown model {model_name}')

    model = model_class(**model_params)

    return model


def append_log_file(filename, log_string):
    with open(filename, 'a') as f:
        f.write(log_string)
        f.write('\n')


@ex.config
def cfg():
    model_name = 'unet'

    patch_size = 0
    make_border = 6

    regularization = 0.000001

    learning_rate = 0.001
    batch_size = 6
    nb_epochs = 50

    use_dice = False


@ex.main
def main(model_name, patch_size, make_border, regularization, learning_rate, batch_size, nb_epochs, use_dice):
    set_variable_repr()

    model_params = {
        'in_channels': 3,
        'out_channels': 1,
    }
    model = create_model(model_name, model_params)

    data_loader_train = create_data_loader(PathologicalImagesDatasetMode.Train, batch_size=batch_size,
                                           patch_size=patch_size, make_border=make_border, augment=True, shuffle=True)
    data_loader_val = create_data_loader(PathologicalImagesDatasetMode.Val, batch_size=batch_size,
                                         patch_size=patch_size, make_border=make_border, augment=True, shuffle=True)

    checkpoint_filename = str(config.MODELS_DIR.joinpath(f'{type(model).__name__}_{patch_size}.ckpt'))
    train_model(model, data_loader_train, data_loader_val, learning_rate, nb_epochs, batch_size, make_border,
                use_dice, regularization, checkpoint_filename)


if __name__ == '__main__':
    ex.run_commandline()
