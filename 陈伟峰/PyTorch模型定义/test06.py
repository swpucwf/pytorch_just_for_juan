import os
import torch
from torchvision import models
model = models.resnet152(pretrained=True)


if __name__ == '__main__':
    # save_dir = "./weights"
    #
    # # 保存整个模型
    # torch.save(model, save_dir)
    # # 保存模型权重
    # torch.save(model.state_dict, save_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # 如果是多卡改成类似0,1,2
    model = model.cuda()  # 单卡
    model = torch.nn.DataParallel(model).cuda()  # 多卡


