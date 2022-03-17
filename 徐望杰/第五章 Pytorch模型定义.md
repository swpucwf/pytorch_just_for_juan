## 5.1 Pytorch模型定义的方式
### 5.1.1 模型的基本结构
- Pytorch定义的模型包含两个基本部分
	- 模型的初始化```__init__```
	- 数据的流向```forward```
- 在Pytorch中定义模型的时候, 我们通过调用torch.nn模块中的Module类来进行神经网络的定义
- 定义方式: 在Module中的forward部分的定义过程中, 可以使用不同的方式来实现神经网络的构建
	- Sequential
	- ModuleList
	- ModuleDict

```
import torch
from torch import nn

class MyNet(nn.Module):
	def __init__(self, ...):
		pass

	def forward(self, x, ...):
		...
		return x
```

### 5.1.2 Sequential
Sequential可以非常简便地定义模型.

它通过接收一个子模块的OrderedDict或者一系列的子模块作为参数来逐个添加Module的实例.

使用Sequential定义的模型不需要写forward, 因为此方法已经内置了顺序的定义. 不过这也导致了Sequential的不够灵活, 没有办法进行模型的参数数量的调整.

```python
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
print(net)

Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
```

### 5.1.3 ModuleList
ModuleList接收的是一个子模块的列表格式作为参数. ModuleList同样具有List的基本性质, 可以进行append, extend等操作, 通过这些方法可以手动给神经网络添加层数以及设置层的权重等参数.

在使用ModuleList的时候需要在```__init__```中先进行定义, 然后在```forward```使用for循环进行调用.

```python
class model(nn.Module):
  def __init__(self, ...):
    self.modulelist = ...
    ...
    
  def forward(self, x):
    for layer in self.modulelist:
      x = layer(x)
    return x
```

### 5.1.4 ModuleDict
ModuleDict和ModuleList的用法和作用都很相似, 不过ModuleDict同样继承了Dict的性质, 可以为神经网络的层进行命名.

```python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
```

## 5.2 利用模型块进行复杂网络的搭建
在5.1中学习到的简单的神经网络的定义和搭建在实际应用场景下不一定具有很好的效果. 在应用场景中, 我们更加频繁地使用更复杂的模型, 具有很大的深度. 在这种情况下, 通过将复杂模型中经常重复出现的层定义为一个模型块, 可以让我们快速的进行模型块的搭建.

现以U-Net为例, 进行模型块的搭建和利用模型块搭建复杂模型的实践.

### 5.2.1 U-Net
U-Net是分割 (Segmentation) 模型的杰作,  在以医学影像为代表的诸多领域有着广泛的应用. U-Net模型结构如下图所示.
![unet](./figures/5.2.1unet.png)

可以通过上图观察到, U-Net具有优秀的对称性.

U-Net的结构中的模型块有:
- 每个子块中的两次卷积(conv)
- 左侧模型块的max pool
- 右侧模块的up-conv
- 输出层

除此之外, 还有模型块之间的copy and crop连接, 这些可以直接通过forward函数实现.

### 5.2.2 U-Net模型块实现
现在我们使用Pytorch可以将上文提到的模型块进行实现, 我们将其命名为: DoubleConv, Down, Up, OutConv.

```python
# 引入依赖包
import torch
import torch.nn as nn
import torch.nn.functional as F
```

```python
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
	    # 继承父类性质
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
		# 进行二次卷积
		self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

```

```python
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

```

```python
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

		# 填充为目标维度的tensor
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

```

```python
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
```

### 5.2.3 U-Net的组装
使用上述的模型块, 就可以很方便地调用它们来组装U-Net模型.

```python
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
	    # 定义每一个模型块的层数
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
```

## 5.3 模型的修改
在我们使用一个现成的模型时, 我们可能会在应用的过程中遇见使用场景和模型结构出现小部分冲突的情况, 在这种情况下学会模型的修改就能起到很大的作用.

### 5.3.1 模型层
在原有的模型中, 可能会出现目标层数与我们实际使用的时候有不同的情况. 

如将ImageNet的最后的fc层的输出节点从1000改为10的情况下, 我们可以定义一个新的fc层进行我们自己的分类, 然后将模型中的fc层定义为我们定义的新的fc层: classifier.
```net.fc = classifier```

通过这种方式, 我们就可以应用新的模型去做自己期望的分类任务.

### 5.3.2 外部输入
在模型训练的过程中, 有时候我们需要输入额外的信息. 如在CNN网络中, 我们有时需要对图像进行补充的信息输入, 此时则需要在神经网络中添加额外的参数.

思路: 在我们期望的添加位置前的部分作为整体, 然后在forward中定义好原模型不变的部分, 以及添加的输入和后续曾的连接关系.

这里一般拼接可以使用```torch.cat```方法, 然后在```__init__```中重新定义后面的部分的层数, 然后在```forward```中进行连接.

### 5.3.3 额外输出
如果我们希望能够在模型的原有基础上输出其中某一中间层的结果, 我怕们可以通过修改forward中的return变量来实现.

## 5.4 模型保存与读取
### 5.4.1 模型储存格式
PyTorch存储模型主要采用pkl，pt，pth三种格式

### 5.4.2 模型储存内容
模型的主要内容:
- 模型结构: 一个继承了nn.Module的Class
- 权重: 一个dict, key为层名, value为该层的权重向量

模型的储存分为两种形式:
- 结构和权重的完全储存```torch.save(model, save_dir)```
- 只储存权重```torch.save(model.state_dict, save_dir)```

### 5.4.3 单卡储存与多卡储存
PyTorch中将模型和数据放到GPU上有两种方式
- .cuda()
- .to(device)

现主要讨论.cuda()的储存方式

使用多卡训练的时候, 需要对模型使用torch.nn.DataParallel
```model = torch.nn.DataParallel(model).cuda()  # 多卡```

### 5.4.4 模型的加载
```python
import os
import torch
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'   #这里替换成希望使用的GPU编号

model = models.resnet152(pretrained=True)
model = nn.DataParallel(model).cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model = nn.DataParallel(loaded_model).cuda()
loaded_model.state_dict = loaded_dict
```