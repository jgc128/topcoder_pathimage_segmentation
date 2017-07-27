import torch
import torch.nn


class CenterCrop2d(torch.nn.Module):
    def __init__(self, size):
        super(CenterCrop2d, self).__init__()

        self.size = size

    def forward(self, inputs):
        if len(inputs.size()) == 4:
            inputs = inputs[:, :, self.size:-self.size, self.size:-self.size]
        else:
            inputs = inputs[:, self.size:-self.size, self.size:-self.size]

        return inputs.contiguous()
