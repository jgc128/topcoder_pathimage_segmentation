import torch
from torch.autograd import Variable


def maybe_to_cuda(obj):
    if torch.cuda.is_available():
        obj = obj.cuda()

    return obj


def set_variable_repr():
    Variable.__repr__ = lambda x: f'Variable {tuple(x.size())}'
