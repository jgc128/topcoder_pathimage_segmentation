import logging
from collections import defaultdict
import copy
import os
import logging

import numpy as np

import torch
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms

from sacred import Experiment
from scipy.stats import gmean
from tqdm import tqdm

import config
from config import MODELS_DIR
from models.fcn import FCN32
from train import create_model, get_checkpoint_filename
from utils.io import save_pickle
from utils.torch.helpers import set_variable_repr, maybe_to_cuda, restore_weights
from utils.torch.datasets import PathologicalImagesDataset, PathologicalImagesDatasetMode, DeterministicPatchesDataset
from utils.torch.transforms import RandomTranspose, RandomHorizontalFlip, RandomVerticalFlip, MakeBorder, CopyNumpy, \
    ImageMaskTransformsCompose, MaskToTensor
from utils.torch.layers import CenterCrop2d

ex = Experiment()


def predict_with_transform(model, data_loader, make_border, target_transforms):
    # save the original transforms to restore later
    original_transforms = copy.deepcopy(data_loader.dataset.transform.transforms)

    for t in target_transforms:
        t.apply_always = True

    target_transforms = ImageMaskTransformsCompose(target_transforms)

    data_loader.dataset.transform.transforms.append(target_transforms)
    data_loader.dataset.transform.transforms.append(CopyNumpy())

    predictions = predict(model, data_loader, make_border)
    predictions = [target_transforms(p, None)[0] for p in predictions]
    predictions = np.array(predictions)

    data_loader.dataset.transform.transforms = original_transforms

    return predictions


def tta_predict(model, data_loader, make_border):
    tta_predictions = []

    needed_transforms = [
        [RandomVerticalFlip(), ],
        [RandomHorizontalFlip(), ],
        [RandomVerticalFlip(), RandomHorizontalFlip(), ],

        # [RandomTranspose(), ],
        # [RandomTranspose(), RandomVerticalFlip(), ],
        # [RandomTranspose(), RandomHorizontalFlip(), ],
        # [RandomTranspose(), RandomVerticalFlip(), RandomHorizontalFlip(), ],
    ]

    # first, no augmentations
    predictions = predict(model, data_loader, make_border)
    tta_predictions.append(predictions)

    for target_transform in needed_transforms:
        predictions = predict_with_transform(model, data_loader, make_border, target_transform)
        tta_predictions.append(predictions)

    tta_predictions = gmean(tta_predictions, axis=0)

    return tta_predictions


def predict(model, data_loader, make_border):
    model = maybe_to_cuda(model)
    model.train(False)

    crop_outputs = None
    if make_border != 0:
        crop_outputs = CenterCrop2d(make_border)

    predictions = []
    tq = tqdm(desc='Prediction', total=len(data_loader.dataset))
    for j, (images, _) in enumerate(data_loader, 1):
        images = torch.autograd.Variable(maybe_to_cuda(images))

        batch_predictions = model(images)
        batch_predictions = batch_predictions.squeeze()

        if crop_outputs is not None:
            batch_predictions = crop_outputs(batch_predictions)

        batch_predictions = F.sigmoid(batch_predictions)

        batch_predictions = batch_predictions.data.cpu().numpy()

        if len(batch_predictions.shape) == 2:
            batch_predictions = np.expand_dims(batch_predictions, 0)

        predictions.append(batch_predictions)
        tq.update(len(images))

    predictions = np.concatenate(predictions)

    return predictions


def create_data_loader(base_dir, mode, nb_folds=5, fold_number=0, batch_size=32, patch_size=224, make_border=0,
                       augment=False):
    transform = []

    if make_border != 0:
        transform.append(MakeBorder(border_size=make_border))

    transform = ImageMaskTransformsCompose(transform)

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
        patch_size, base_dir, mode=mode, nb_folds=nb_folds, fold_number=fold_number,
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

        if patch_size != 0:
            c_h = (patch_coord[0], patch_coord[0] + patch_size)
            c_w = (patch_coord[1], patch_coord[1] + patch_size)

            predictions[image_idx, c_h[0]:c_h[1], c_w[0]:c_w[1]] += patch_pred
            nb_preds[image_idx, c_h[0]:c_h[1], c_w[0]:c_w[1]] += 1
        else:
            predictions[image_idx, :, :] += patch_pred
            nb_preds[image_idx, :, :] += 1

    # TODO: geometric mean
    predictions = predictions / nb_preds

    return predictions


def save_predictions(filename, images, predictions):
    to_save = [images, predictions]
    save_pickle(filename, to_save)


def get_prediction_filename(model_name, mode, patch_size_train, patch_size_predict, fold_number, use_dice):
    use_dice = int(use_dice)
    prediction_filename = config.PREDICTIONS_DIR.joinpath(
        'folds/',
        f'{model_name}_patch{patch_size_train}_predict{patch_size_predict}_fold{fold_number}_dice{use_dice}_{mode.name.lower()}.pkl'
    )
    return prediction_filename


@ex.config
def cfg():
    model_name = 'unet'
    patch_size_train = 0
    patch_size_predict = 0
    make_border = 6
    nb_folds = 5
    fold_number = 0

    use_dice = False

    batch_size = 2


@ex.main
def main(model_name, patch_size_train, patch_size_predict, make_border, nb_folds, fold_number, use_dice, batch_size):
    set_variable_repr()

    model_params = {
        'in_channels': 3,
        'out_channels': 1,
    }
    model = create_model(model_name, model_params)
    logging.info('Model created')

    checkpoint_filename = str(get_checkpoint_filename(model_name, patch_size_train, fold_number, use_dice))
    restore_weights(model, checkpoint_filename)

    configurations = [
        {'mode': PathologicalImagesDatasetMode.Val, 'base_dir': config.DATASET_TRAIN_DIR, },
        {'mode': PathologicalImagesDatasetMode.Train, 'base_dir': config.DATASET_TRAIN_DIR, },
        {'mode': PathologicalImagesDatasetMode.All, 'base_dir': config.DATASET_TEST_DIR, },
    ]
    for conf in configurations:
        mode = conf['mode']
        base_dir = conf['base_dir']
        predictions_filename = get_prediction_filename(model_name, mode, patch_size_train, patch_size_predict,
                                                       fold_number, use_dice)

        data_loader = create_data_loader(base_dir=base_dir, mode=mode, nb_folds=nb_folds, fold_number=fold_number,
                                         batch_size=batch_size,
                                         patch_size=patch_size_predict, make_border=make_border, augment=False)

        # patches_predictions = predict(model, data_loader, make_border=make_border)
        patches_predictions = tta_predict(model, data_loader, make_border=make_border)
        logging.info(f'Patch predictions: {patches_predictions.shape}')

        predictions = combine_patches(patches_predictions, data_loader.dataset.patches,
                                      data_loader.dataset.nb_images, data_loader.dataset.image_height,
                                      data_loader.dataset.image_width, data_loader.dataset.patch_size)
        logging.info(f'Predictions: {predictions.shape}')

        save_predictions(predictions_filename, data_loader.dataset.images, predictions)
        logging.info(f'Predictions saved: {predictions_filename}')


if __name__ == '__main__':
    ex.run_commandline()
