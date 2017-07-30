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
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from tqdm import tqdm

import config
from models import FCN32, UNet, Tiramisu
from predict import get_prediction_filename
from train import create_model, train_model
from utils.io import load_pickle, load_image
from utils.torch.helpers import set_variable_repr, maybe_to_cuda
from utils.torch.datasets import PathologicalImagesDataset, PathologicalImagesDatasetMode, TransformDataset
import utils.torch.transforms
from utils.torch.layers import CenterCrop2d
from utils.torch.losses import DiceWithLogitsLoss

ex = Experiment()


def load_data(model_name, patch_size_train, patch_size_predict, nb_folds, fold_number, use_dice):
    # load predictions for the models
    folds_train = []
    folds_val = []
    masks_train = []
    masks_val = []
    for fold_idx in range(nb_folds):
        filename = get_prediction_filename(model_name, PathologicalImagesDatasetMode.Val, patch_size_train,
                                           patch_size_predict, fold_idx, use_dice)
        images, predictions = load_pickle(filename)

        # load masks
        masks_filenames = [
            image.parent.parent.joinpath('truth/').joinpath(image.stem + '_mask.png')
            for image in images
        ]
        masks = [load_image(filename, grayscale=True) for filename in masks_filenames]

        # target fold for val
        if fold_idx == fold_number:
            folds_val.append(predictions)
            masks_val.append(masks)
        else:
            folds_train.append(predictions)
            masks_train.append(masks)

    data_train = np.concatenate(folds_train, axis=0)
    data_val = np.concatenate(folds_val, axis=0)
    masks_train = np.concatenate(masks_train, axis=0)
    masks_val = np.concatenate(masks_val, axis=0)

    logging.info(f'Data train: {data_train.shape}, {masks_train.shape}')
    logging.info(f'Data val: {data_val.shape}, {masks_val.shape}')

    return data_train, masks_train, data_val, masks_val


def create_data_loader(images, masks, make_border, augment=True, shuffle=True, batch_size=32):
    transform = []

    if make_border != 0:
        transform.append(utils.torch.transforms.MakeBorder(border_size=make_border))

    if augment:
        transform.extend([
            # utils.torch.transforms.RandomTranspose(),
            utils.torch.transforms.RandomVerticalFlip(),
            utils.torch.transforms.RandomHorizontalFlip(),
            utils.torch.transforms.CopyNumpy(),
        ])

    transform = utils.torch.transforms.ImageMaskTransformsCompose(transform)

    image_transform = [
        utils.torch.transforms.UnSqueezeChannel(),
        utils.torch.transforms.ToTensor(transpose=True),
    ]
    image_transform = torchvision.transforms.Compose(image_transform)

    mask_transform = [
        utils.torch.transforms.ToTensor(divide=True)
    ]
    mask_transform = torchvision.transforms.Compose(mask_transform)

    data_set = TransformDataset(images, masks, transform=transform, image_transform=image_transform,
                                mask_transform=mask_transform)

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=shuffle, num_workers=1,
                                              pin_memory=torch.cuda.is_available())

    return data_loader


def get_checkpoint_filename(model_name, patch_size_train, patch_size_predict, fold_number, use_dice):
    use_dice = int(use_dice)
    checkpoint_filename = config.MODELS_DIR.joinpath(
        f'corrector_{model_name}_patch{patch_size_train}_predict{patch_size_predict}_fold{fold_number}_dice{use_dice}.ckpt'
    )
    return checkpoint_filename


@ex.config
def cfg():
    model_name = 'unet'

    patch_size_train = 0
    patch_size_predict = 0
    make_border = 6

    nb_folds = 5
    fold_number = 0
    use_dice = False

    regularization = 0.000001
    learning_rate = 0.001
    batch_size = 4
    nb_epochs = 100


@ex.main
def main(model_name, patch_size_train, patch_size_predict, make_border, nb_folds, fold_number, regularization,
         learning_rate, batch_size, nb_epochs, use_dice):
    set_variable_repr()

    set_variable_repr()

    model_params = {
        'in_channels': 1,
        'out_channels': 1,
    }
    model = create_model(model_name, model_params)

    images_train, masks_train, images_val, masks_val = load_data(model_name, patch_size_train, patch_size_predict,
                                                                 nb_folds,
                                                                 fold_number, use_dice)

    data_loader_train = create_data_loader(images_train, masks_train, make_border, augment=True, shuffle=True,
                                           batch_size=batch_size)

    data_loader_val = create_data_loader(images_val, masks_val, make_border, augment=True, shuffle=True,
                                         batch_size=batch_size)

    checkpoint_filename = get_checkpoint_filename(model_name, patch_size_train, patch_size_predict, fold_number,
                                                  use_dice)
    train_model(model, data_loader_train, data_loader_val, learning_rate, nb_epochs, batch_size, make_border,
                use_dice, regularization, checkpoint_filename=str(checkpoint_filename), log_filename=None)


if __name__ == '__main__':
    ex.run_commandline()
