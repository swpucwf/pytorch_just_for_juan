教程前四章的打卡记录：[深入迁出pytorch专栏](https://blog.csdn.net/soraca/category_11409115.html)
该教程的GitHub地址：[深入浅出PyTorch
](https://github.com/datawhalechina/thorough-pytorch)哔哩哔哩视频地址：[深入浅出Pytorch](https://www.bilibili.com/video/BV1e341127Lt?p=1)

# DataWhale“深入浅出PyTorch”第六章—— PyTorch进阶训练技巧
@[toc]
## 1. 自定义损失函数
**现状及重要性**：
- 官方损失函数：的PyTorch在torch.nn模块为我们提供了许多常用的损失函数，比如：MSELoss，L1Loss，BCELoss...... 
- 非官方的损失函数：随着深度学习的发展，出现了越来越多的非官方提供的Loss，比如DiceLoss，HuberLoss，SobolevLoss......这些Loss Function专门针对一些非通用的模型，PyTorch不能将他们全部添加到库中去，因此这些损失函数的实现则需要我们通过自定义损失函数来实现。
-  新的损失函数：另外，在科学研究中，我们往往会提出全新的损失函数来提升模型的表现，这时我们既无法使用PyTorch自带的损失函数，也没有相关的博客供参考，此时自己实现损失函数就显得更为重要了。

### 1.1 以函数的方式定义
直接以函数定义的方式定义一个自己的函数，例如下面的平方损失函数：

```python
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```
### 1.2 以类的方式定义
虽然以函数定义的方式很简单，但是以类方式定义更加常用，在以类方式定义损失函数时，我们如果看每一个损失函数的继承关系我们就可以发现`Loss`函数部分继承自`_loss`, 部分继承自`_WeightedLoss`, 而`_WeightedLoss`继承自`_loss`，` _loss`继承自 **nn.Module**。下面以**DiceLoss**为例。
Dice Loss是一种在分割领域常见的损失函数，定义如下：


$DSC = \frac{2|X∩Y|}{|X|+|Y|}$


代码实现如下：

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

# 使用方法    
criterion = DiceLoss()
loss = criterion(input,targets)

```
其余常见函数BCE-Dice Loss ,IoU Loss,Focal Loss的定义与使用可见[链接](https://github.com/datawhalechina/thorough-pytorch/blob/main/%E7%AC%AC%E5%85%AD%E7%AB%A0%20PyTorch%E8%BF%9B%E9%98%B6%E8%AE%AD%E7%BB%83%E6%8A%80%E5%B7%A7/6.1%20%E8%87%AA%E5%AE%9A%E4%B9%89%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.md)。


## 2. 动态调整学习率
**调整学习率的重要性：**
- 学习速率过小：会极大降低收敛速度，增加训练时间
- 学习速率过大：可能导致参数在最优解两侧来回振荡，导致无法收敛


当我们选定了一个合适的学习率后，经过许多轮的训练后，可能会出现准确率震荡或loss不再下降等情况，说明当前学习率已不能满足模型调优的需求。此时我们就可以通过一个适当的学习率衰减策略是一个来改善这种现象，提高我们的精度。这种设置方式在PyTorch中被称为**scheduler**。

### 2.1 使用官方scheduler
PyTorch在`torch.optim.lr_scheduler`封装好的一些动态调整学习率的方法。下面是列出的这些scheduler。
+ [`lr_scheduler.LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR)
+ [`lr_scheduler.MultiplicativeLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR)
+ [`lr_scheduler.StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR)
+ [`lr_scheduler.MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)
+ [`lr_scheduler.ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR)
+ [`lr_scheduler.CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)
+ [`lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
+ [`lr_scheduler.CyclicLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
+ [`lr_scheduler.OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)
+ [`lr_scheduler.CosineAnnealingWarmRestarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)


**使用上面的官方API的方法：**

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
**注意**：我们在使用官方给出的`torch.optim.lr_scheduler`时，需要将`scheduler.step()`放在`optimizer.step()`后面进行使用。


### 2.2 自定义scheduler
虽然PyTorch官方给我们提供了许多的API，但是在实验中我们也有可能碰到需要我们自己定义学习率调整策略的情况，而我们的方法是自定义函数`adjust_learning_rate`来改变`param_group`中`lr`的值.

假设我们需要我们的学习率每30轮下降为原来的1/10且已有的官方API中没有符合我们需求的，然后自定义函数来实现学习率的改变。

代码如下：

```python
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))#关键代码，实现每30轮下降为原来的1/10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```


## 3. 模型微调
~~没看懂该如何实验先搁置~~ 

## 4.半精度训练
**现实依据：**
GPU的性能主要分为两部分：**算力和显存**，前者决定了显卡计算的速度，后者则决定了显卡可以同时放入多少数据用于计算。在可以使用的显存数量一定的情况下，每次训练能够加载的数据更多（也就是batch size更大），则也可以提高训练效率。**即显存大小决定能否训练，算力决定了训练的快慢，重要性显存>算力**
另外，有时候数据本身也比较大（比如3D图像、视频等），显存较小的情况下可能甚至batch size为1的情况都无法实现。因此，合理使用显存也就显得十分重要。

**理论依据:**
PyTorch默认的浮点数存储方式用的是torch.float32，小数点后位数更多固然能保证数据的精确性，但绝大多数场景其实并不需要这么精确，**只保留一半的信息也不会影响结果，也就是使用torch.float16格式**。由于数位减了一半，因此被称为“半精度”，具体如下图：
~~图的文章出处找不到了~~ 
![在这里插入图片描述](https://img-blog.csdnimg.cn/f7b54fc2c7754eeea85f6db28c794845.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
显然半精度能够减少显存占用，使得显卡可以同时加载更多数据进行计算。
PyTorch中使用autocast函数配置半精度训练，同时需要在三处加以设置以完成半精度训练。

### 4.1 半精度训练的设置
**1.导包 import autocast**

```python
from torch.cuda.amp import autocast
```
**2.模型设置**
在模型定义中，使用python的装饰器方法，用autocast装饰模型中的forward函数。关于装饰器的使用，可以参考[这里](https://www.cnblogs.com/jfdwd/p/11253925.html)：
```python
@autocast()   
def forward(self, x):
    ...
    return x
```
**3.训练过程**
在训练过程中，只需在将数据输入模型及其之后的部分放入“with autocast():“

```python
 for x in train_loader:
	x = x.cuda()
	with autocast():
        output = model(x)
        ...
```

**注意：**

半精度训练主要适用于数据本身的size比较大（比如说3D图像、视频等）。当数据本身的size并不大时（比如手写数字MNIST数据集的图片尺寸只有28*28），使用半精度训练则可能不会带来显著的提升。

