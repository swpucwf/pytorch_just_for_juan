import torch
from torch import nn
import torch.nn.functional as f


class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
        self.ModuleDict = args

    def forward(self, x):
        for key,value in self.ModuleDict.items():
            print("当前层：",key)
            print("当前输出",value(x).shape)
            x = value(x)
        # for m in self.ModuleDict.values():
        #     x = m(x)
        return x


if __name__ == '__main__':
    # key-value
    args = nn.ModuleDict({
        'linear': nn.Linear(784, 256),
        'act': nn.ReLU()
    })
    args['output'] = nn.Linear(256, 10)
    # 添加

    net = model(args)




    x = torch.randn(1,784)
    print(net(x).shape)