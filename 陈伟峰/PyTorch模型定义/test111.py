from torchvision.models import resnet50


if __name__ == '__main__':
    model = resnet50()
    # for layer in model.children():
    #     print(layer)
    print(model._modules.items())