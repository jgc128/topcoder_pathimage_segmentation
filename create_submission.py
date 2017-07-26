import logging
from collections import defaultdict
import shutil

import os
import logging

import numpy as np

from sacred import Experiment

import config
from utils.io import load_pickle

ex = Experiment()


def create_submission_files(submission_dir, images, predictions, threshold):
    submission_dir.mkdir(exist_ok=True)

    for image, pred in zip(images, predictions):
        pred_filename = submission_dir.joinpath(f'{image.stem}_mask.txt')

        pred_mask = np.zeros_like(pred, dtype=np.uint8)
        pred_mask[pred > threshold] = 1
        np.savetxt(pred_filename, np.transpose(pred_mask, (1, 0)), fmt='%d', delimiter='')

    logging.info(f'Submission files created: {submission_dir}, {len(images)}')


@ex.config
def cfg():
    model_class = 'UNet'
    patch_size_train = 480
    patch_size_predict = 480
    threshold = 0.4


@ex.main
def main(model_class, patch_size_train, patch_size_predict, threshold):
    submission_dir = config.SUBMISSIONS_DIR.joinpath(
        f'{model_class}_{patch_size_train}_{patch_size_predict}_{threshold}'
    )

    filename_test = config.PREDICTIONS_DIR.joinpath(f'{model_class}_{patch_size_train}_{patch_size_predict}_test.pkl')
    images_test, predictions_test = load_pickle(filename_test)
    create_submission_files(submission_dir, images_test, predictions_test, threshold)

    filename_train = config.PREDICTIONS_DIR.joinpath(f'{model_class}_{patch_size_train}_{patch_size_predict}_train.pkl')
    images_train, predictions_train = load_pickle(filename_train)
    create_submission_files(submission_dir, images_train, predictions_train, threshold)

    # create archive
    submission_filename = shutil.make_archive(submission_dir, 'zip', submission_dir)
    logging.info(f'Archive created: {submission_filename}')


if __name__ == '__main__':
    ex.run_commandline()
