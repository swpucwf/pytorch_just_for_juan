import numpy as np
import torch

def MSE(target,output):
    if not isinstance(target,torch.Tensor)  or not isinstance(output,torch.Tensor):
        target = torch.tensor(target)
        output = torch.tensor(output)
    return ((output-target)**2).sum().mean()




if __name__ == '__main__':
    target = torch.randn(20)
    output = torch.randn(20)
    print(MSE(target,output))