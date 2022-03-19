# **1. 自定义损失函数**

Pytorch的库中自带了许多常用的损失函数，比如：MES，BCELoss等。但是随着深度学习等发展，出现了越来越的Loss，并且这些loss是专门针对一些非通用模型的，所以pytorch没有将这些loss加入库中。若是自己要利用这些loss进行训练则需要以自定义的方式实现这些损失函数。这里实现自定义的损失函数有两种方式：函数和类。



### **1.1函数方式定义损失函数**

```python
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```

利用函数式损失函数可以直接计算模型输出和label之间的差距



### **1.2 类方式定义损失函数**

```python
class DiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()
        
	def forward(self,inputs,targets,smooth=1)
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice
```

pytorch一般都是以类来定义损失函数，继承nn.Module。利用类定义损失函数有一个好处，就是可以利用cuda通过GPU计算。



# **2.动态调整学习率**

在训练过程中，学习率是一个很重要的参数。学习率调整的大可能学得快但是不一定可以拟合，会在最优解附近震荡。如果是我们选择一个合适的学习率，可能会出现loss下降到某给点就不再下降，这时候可以通过一个学习率衰减来改善这种现象，提高精度。这种设置方式在pytorch中被称为scheduler。



### **2.1 官方api**

```python
# 选择一种优化器
optimizer = torch.optim.Adam(...) 
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 需要在优化器参数更新之后再动态调整学习率
	scheduler1.step() 
	...
    schedulern.step()
```

值得**注意**的是这里我们需要将`scheduler.step()`放在`optimizer.step()`后进行使用。



### **2.2 自定义scheduler**

假设我们需要自己定义学习率调整策略，需要自己改变`param_group`中的`lr`值，我们可以

```python
# 每30个epoch，将学习率设置为当前的1/10
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30)) # 用幂的方式更新
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

