from torch import nn
import torch
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)

    def forward(self, x, add_variable):

        x = self.net(x)

        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)), 1)

        x = self.fc_add(x)

        x = self.output(x)

        return x
if __name__ == '__main__':
    import torchvision.models as models
    net = models.resnet50()
    model = Model(net)# .cuda()
    x = torch.randn(1,3,224,224)

    print(model(x,torch.Tensor([1])).shape)
