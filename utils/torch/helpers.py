import os
import logging

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
