import torch
from torch import nn


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        self.ModuleList = args

    def forward(self, x):
        for m in self.ModuleList:
            x = m(x)
        return x


if __name__ == '__main__':
    args = nn.ModuleList([
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 10)
    ])
    net = model(args)
    print(net)
    # net1(
    #   (modules): ModuleList(
    #     (0): Linear(in_features=10, out_features=10, bias=True)
    #     (1): Linear(in_features=10, out_features=10, bias=True)
    #   )
    # )

    for param in net.parameters():
        print(type(param.data), param.size())

    # class 'torch.Tensor'> torch.Size([256, 784])
    # class 'torch.Tensor'> torch.Size([256])
    # class 'torch.Tensor'> torch.Size([10, 256])
    # class 'torch.Tensor'> torch.Size([10])
    x = torch.randn(1,784)
    print(net)
    print(net(x).shape)