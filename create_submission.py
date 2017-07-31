import logging
from collections import defaultdict
import shutil

import os
import logging

import numpy as np

from sacred import Experiment
from scipy.stats import gmean

import config
from predict import get_prediction_filename
from utils.io import load_pickle
from utils.torch.datasets import PathologicalImagesDatasetMode
from utils.postprocessing import morph_masks

ex = Experiment()


def create_submission_files(submission_dir, images, predictions, threshold, morph_open, morph_close):
    submission_dir.mkdir(exist_ok=True)

    for image, pred in zip(images, predictions):
        pred_filename = submission_dir.joinpath(f'{image.stem}_mask.txt')

        pred_mask = np.zeros_like(pred, dtype=np.uint8)
        pred_mask[pred > threshold] = 1

        if morph_open != 0:
            pred_mask = morph_masks(pred_mask, kernel_size=morph_open, operation='open')

        if morph_close != 0:
            pred_mask = morph_masks(pred_mask, kernel_size=morph_close, operation='close')

        np.savetxt(pred_filename, np.transpose(pred_mask, (1, 0)), fmt='%d', delimiter='')

    logging.info(f'Submission files created: {submission_dir}, {len(images)}')


def get_submission_dir_name(model_name, patch_size_train, patch_size_predict, threshold, use_dice, use_tta, morph_open,
                            morph_close):
    use_dice = int(use_dice)
    use_tta = int(use_tta)
    threshold = str(threshold).replace('.', '')

    submission_dir_name = config.SUBMISSIONS_DIR.joinpath(
        'folds/',
        f'{model_name}_patch{patch_size_train}_predict{patch_size_predict}'
        f'_dice{use_dice}_tta{use_tta}_tr{threshold}_open{morph_open}_close{morph_close}'
    )
    return submission_dir_name


@ex.config
def cfg():
    model_name = 'unet_ds'
    patch_size_train = 0
    patch_size_predict = 0

    nb_folds = 5
    use_dice = False
    use_tta = False

    threshold = 0.3
    average_mode = 'gmean'
    morph_open = 0
    morph_close = 0


@ex.main
def main(model_name, patch_size_train, patch_size_predict, nb_folds, use_dice, use_tta, threshold, average_mode,
         morph_open, morph_close):
    submission_dir = get_submission_dir_name(model_name, patch_size_train, patch_size_predict, threshold, use_dice,
                                             use_tta, morph_open, morph_close)

    configurations = [
        {'mode': PathologicalImagesDatasetMode.Val, 'base_dir': config.DATASET_TRAIN_DIR, },
        {'mode': PathologicalImagesDatasetMode.Train, 'base_dir': config.DATASET_TRAIN_DIR, },
        {'mode': PathologicalImagesDatasetMode.All, 'base_dir': config.DATASET_TEST_DIR, },
    ]
    for conf in configurations:
        mode = conf['mode']
        base_dir = conf['base_dir']

        predictions = defaultdict(list)

        for fold_number in range(nb_folds):
            predictions_filename = get_prediction_filename(model_name, mode, patch_size_train, patch_size_predict,
                                                           fold_number, use_dice, use_tta)
            images, fold_predictions = load_pickle(predictions_filename)

            for image, image_pred in zip(images, fold_predictions):
                predictions[image].append(image_pred)

        # get mean of all folds
        images = sorted(predictions.keys())
        logging.info(f'Average mode: {average_mode}')
        if average_mode == 'gmean':
            predictions = [gmean(predictions[image]) for image in images]
        elif average_mode == 'mean':
            predictions = [np.mean(predictions[image], axis=0) for image in images]
        else:
            ValueError(f'Average mode {average_mode} unknown')

        create_submission_files(submission_dir, images, predictions, threshold, morph_open, morph_close)

    # create archive
    submission_filename = shutil.make_archive(submission_dir, 'zip', submission_dir)
    logging.info(f'Archive created: {submission_filename}')


if __name__ == '__main__':
    ex.run_commandline()
