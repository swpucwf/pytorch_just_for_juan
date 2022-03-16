import torchvision.models as models
from torch import nn
from collections import OrderedDict




if __name__ == '__main__':
    # 原始模型
    net = models.resnet50()

    classifier = nn.Sequential(OrderedDict(
        [('fc1', nn.Linear(2048, 128)),
         ('relu1', nn.ReLU()),
         ('dropout1', nn.Dropout(0.5)),
         ('fc2', nn.Linear(128, 10)),
         ('output', nn.Softmax(dim=1))
         ]))

    net.fc = classifier

    print(net)
