import os
import torch
from torchvision import models


if __name__ == '__main__':
    model = models.resnet50(pretrained=True)
    print(model)
    save_dir = "test.pt"
    model.cuda()

    torch.save(model,save_dir)

    loaded_mode= torch.load(save_dir)

    loaded_mode.cuda()

    #
    torch.save(model.state_dict(),save_dir)
    loaded_dict = torch.load(save_dir)
    loaded_model = models.resnet50()

    # loaded_model.load_state_dict(loaded_dict)
    # loaded_model.state_dict = loaded_dict



    loaded_model.cuda()
