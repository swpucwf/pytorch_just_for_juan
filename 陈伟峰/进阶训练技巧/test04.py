import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1,isMin=False):
        '''

        :param inputs: box
        :param targets: boxes
        :param smooth: 1
        :param isMin: False
        :return:
        '''
        inputs_area = (inputs[3] - inputs[1]) * (inputs[2] - inputs[0])
        targets_area = (targets[:, 3] - targets[:, 1]) * (targets[:, 2] - targets[:, 0])

        xx1 = torch.maximum(inputs[0], targets[:, 0])
        yy1 = torch.maximum(inputs[1], targets[:, 1])
        xx2 = torch.minimum(inputs[2], targets[:, 2])
        yy2 = torch.minimum(inputs[3], targets[:, 3])

        w = torch.maximum(torch.zeros(1), (xx2 - xx1))
        h = torch.maximum(torch.zeros(1), (yy2 - yy1))

        area = w * h

        if isMin:
            return torch.true_divide(area+smooth, torch.minimum(inputs_area, targets_area)+smooth)
        else:
            return torch.true_divide(area+smooth, inputs_area + targets_area - area+smooth)

if __name__ == '__main__':
    loss= IoULoss()
    a = torch.tensor([1, 1, 11, 11])
    bs = torch.tensor([[1, 1, 10, 10], [11, 11, 20, 20]])
    print(loss(a, bs))
