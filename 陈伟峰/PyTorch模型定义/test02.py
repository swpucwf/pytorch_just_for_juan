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

        # x
        # nsv
        # nchw
        # n,c,t,h,w

        # reshape   ------> nchw ----> nsv
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        # 形状变换。conv + bn + relu + 。。。。。。 + avgpool or nn.Linear()

        # 10,softmax
        temp = None
        for name,value in self._modules.items():
            # self._modules.values 保存的是key-value
            print(name)
            print(value)
            print(value[0])
            x = value(x)
            if name=="0":
                temp = value[0](x)


        #
        return temp,x





if __name__ == '__main__':

    args = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    model = MySequential(args)
    x = torch.randn(1,784)

    # print(model(x))
    # print(model.forward(x))
    # print(model())
    print(model(x))
    # shape
    # maxpooling