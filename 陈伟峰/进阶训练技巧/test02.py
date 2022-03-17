import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


if __name__ == '__main__':
    inputs = torch.ones(224,224)
    targets = torch.zeros(224,224)
    # 使用方法
    criterion = DiceLoss()
    loss = criterion(inputs, targets)
    print(loss)