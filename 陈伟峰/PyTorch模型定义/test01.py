import torch
from torch import nn
from collections import OrderedDict


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, x):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            # self._modules.values 保存的是key-value
            print(module)
            x = module(x)
        return x


if __name__ == '__main__':
    args = OrderedDict([
                  ('conv1', nn.Conv2d(3,20,(5,5))),
                  ('relu1', nn.ReLU()),
                  ('conv2', nn.Conv2d(20,64,(5,5))),
                  ('relu2', nn.ReLU())
                ])
    model = MySequential(args)
    x = torch.randn(1,3,224,224)
    print(model)
    print(model(x).shape)
