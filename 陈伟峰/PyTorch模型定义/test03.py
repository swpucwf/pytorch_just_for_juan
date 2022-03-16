import torch
from torch import nn


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        self.ModuleList = args

    def forward(self, x):
        # for m in self.ModuleList:
        #     x = m(x)
        #
        x = self.ModuleList[0](x)
        self.ModuleList.append(  nn.Linear(10, 784))

        x =  self.ModuleList[2](x)
        # print(x.shape)
        x = self.ModuleList[3](x)
        # print(x.shape)
        x =  self.ModuleList[0](x)
        x =  self.ModuleList[2](x)

        return x


if __name__ == '__main__':

    # for i in list():
    #     pass
    args = nn.ModuleList([
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ])


    for i in range(8):
        self.model.append(ResNet)


    net = model(args)
    # print(net)
    x = torch.randn(1,784)

    print(net(x).shape)

    # net1(
    #   (modules): ModuleList(
    #     (0): Linear(in_features=10, out_features=10, bias=True)
    #     (1): Linear(in_features=10, out_features=10, bias=True)
    #   )
    # )

    # for param in net.parameters():
    #     print(type(param.data), param.size())
    #
    # # class 'torch.Tensor'> torch.Size([256, 784])
    # # class 'torch.Tensor'> torch.Size([256])
    # # class 'torch.Tensor'> torch.Size([10, 256])
    # # class 'torch.Tensor'> torch.Size([10])
    # x = torch.randn(1,784)
    # print(net)
    # print(net(x).shape)