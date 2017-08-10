import os
import logging

import numpy as np

import torch
from torch.autograd import Variable


def maybe_to_cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()

    return obj


def set_variable_repr():
    Variable.__repr__ = lambda x: f'Variable {tuple(x.size())}'


def restore_weights(model, filename):
    map_location = None

    # load trained on GPU models to CPU
    if not torch.cuda.is_available():
        map_location = lambda storage, loc: storage

    state_dict = torch.load(filename, map_location=map_location)

    model.load_state_dict(state_dict)

    logging.info(f'Model restored {os.path.basename(filename)}')


def cyclic_lr_scheduler(optimizer, iteration, epoch, base_lr=0.001, max_lr=0.006, step_size=100):
    # TODO: move to parameters
    if epoch > 200:
        base_lr /= 10
        max_lr /= 10

    if epoch > 300:
        base_lr /= 10
        max_lr /= 10

    if epoch > 350:
        base_lr /= 10
        max_lr /= 10

    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs(iteration / step_size - 2 * cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer
