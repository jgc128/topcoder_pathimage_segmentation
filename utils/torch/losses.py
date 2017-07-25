import torch
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable


class DiceWithLogitsLoss(torch.nn.Module):
    def __init__(self):
        super(DiceWithLogitsLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs.view(-1))
        targets = targets.view(-1)

        epsilon = 1e-5
        denominator = inputs.sum() + targets.sum() + epsilon
        numerator = 2 * torch.sum(inputs * targets) + epsilon

        dice = numerator / denominator

        loss = 1 - dice

        return loss
