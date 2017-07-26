import logging
from collections import defaultdict

import os
import logging

import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms

from sacred import Experiment
from tqdm import tqdm

import config
from config import MODELS_DIR
from models.fcn import FCN32
from train import create_model
from utils.io import save_pickle
from utils.torch.helpers import set_variable_repr, maybe_to_cuda, restore_weights
from utils.torch.datasets import PathologicalImagesDataset, PathologicalImagesDatasetMode, DeterministicPatchesDataset
from utils.torch.transforms import MaskToTensor, ImageMaskTransformsCompose, SamplePatch, RandomTranspose, \
    RandomVerticalFlip, RandomHorizontalFlip, CopyNumpy

ex = Experiment()


def predict(model, data_loader):
    model = maybe_to_cuda(model)
    model.train(False)

    predictions = []
    tq = tqdm(desc='Prediction', total=len(data_loader.dataset))
    for j, (images, _) in enumerate(data_loader, 1):
        images = torch.autograd.Variable(maybe_to_cuda(images))

        batch_predictions = model(images)
        batch_predictions = batch_predictions.squeeze()
        batch_predictions = F.sigmoid(batch_predictions)

        predictions.append(batch_predictions.data.cpu().numpy())
        tq.update(len(images))

    predictions = np.concatenate(predictions)

    return predictions


def create_data_loader(mode, base_dir, batch_size=32, patch_size=224, augment=False):
    transform = None

    image_transform = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.70500564, 0.4902217, 0.6467339], std=[0.19247672, 0.20918619, 0.15601342]
        ),
    ]
    image_transform = torchvision.transforms.Compose(image_transform)

    mask_transform = [
        MaskToTensor()
    ]
    mask_transform = torchvision.transforms.Compose(mask_transform)

    data_set = DeterministicPatchesDataset(
        patch_size, base_dir, mode=mode,
        transform=transform, image_transform=image_transform, mask_transform=mask_transform
    )
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=1,
                                              pin_memory=torch.cuda.is_available())

    return data_loader


def combine_patches(patches_predictions, patches, nb_images, image_height, image_width, patch_size):
    predictions = np.zeros((nb_images, image_height, image_width), dtype=np.float32)
    nb_preds = np.zeros((nb_images, image_height, image_width), dtype=np.float32)

    for patch_pred, patch_info in zip(patches_predictions, patches):
        image_idx = patch_info[0]
        patch_coord = patch_info[1]

        c_h = (patch_coord[0], patch_coord[0] + patch_size)
        c_w = (patch_coord[1], patch_coord[1] + patch_size)

        predictions[image_idx, c_h[0]:c_h[1], c_w[0]:c_w[1]] += patch_pred
        nb_preds[image_idx, c_h[0]:c_h[1], c_w[0]:c_w[1]] += 1

    predictions = predictions / nb_preds

    return predictions


def save_predictions(filename, images, predictions):
    to_save = [images, predictions]
    save_pickle(filename, to_save)


def predict_and_save(model, base_dir, mode, predictions_filename, batch_size, patch_size):
    data_loader = create_data_loader(mode=mode, base_dir=base_dir, batch_size=batch_size, patch_size=patch_size,
                                     augment=False)

    patches_predictions = predict(model, data_loader)
    logging.info(f'Patch predictions: {patches_predictions.shape}')

    predictions = combine_patches(patches_predictions, data_loader.dataset.patches,
                                  data_loader.dataset.nb_images, data_loader.dataset.image_height,
                                  data_loader.dataset.image_width, data_loader.dataset.patch_size)
    logging.info(f'Predictions: {predictions.shape}')

    save_predictions(predictions_filename, data_loader.dataset.images, predictions)
    logging.info(f'Predictions saved: {predictions_filename}')


@ex.config
def cfg():
    model_name = 'unet'
    patch_size_train = 480
    patch_size_predict = 480
    batch_size = 10


@ex.main
def main(model_name, patch_size_train, patch_size_predict, batch_size):
    set_variable_repr()

    model_params = {
        'in_channels': 3,
        'out_channels': 1,
    }
    model = create_model(model_name, model_params)
    logging.info('Model created')

    checkpoint_filename = str(MODELS_DIR.joinpath(f'{type(model).__name__}_{patch_size_train}.ckpt'))
    restore_weights(model, checkpoint_filename)

    # predict val
    base_dir_val = config.DATASET_TRAIN_DIR
    mode_val = PathologicalImagesDatasetMode.Val
    predictions_filename_val = config.PREDICTIONS_DIR.joinpath(
        f'{type(model).__name__}_{patch_size_train}_{patch_size_predict}_val.pkl'
    )
    predict_and_save(model, base_dir_val, mode_val, predictions_filename_val, batch_size, patch_size_predict)

    # predict on the whole train
    base_dir_train = config.DATASET_TRAIN_DIR
    mode_train = PathologicalImagesDatasetMode.All
    predictions_filename_train = config.PREDICTIONS_DIR.joinpath(
        f'{type(model).__name__}_{patch_size_train}_{patch_size_predict}_train.pkl'
    )
    predict_and_save(model, base_dir_train, mode_train, predictions_filename_train, batch_size, patch_size_predict)

    # predict test
    base_dir_test = config.DATASET_TEST_DIR
    mode_test = PathologicalImagesDatasetMode.All
    predictions_filename_test = config.PREDICTIONS_DIR.joinpath(
        f'{type(model).__name__}_{patch_size_train}_{patch_size_predict}_test.pkl'
    )
    predict_and_save(model, base_dir_test, mode_test, predictions_filename_test, batch_size, patch_size_predict)


if __name__ == '__main__':
    ex.run_commandline()
