教程前四章的打卡记录：[深入迁出pytorch专栏](https://blog.csdn.net/soraca/category_11409115.html)
该教程的GitHub地址：[深入浅出PyTorch
](https://github.com/datawhalechina/thorough-pytorch)哔哩哔哩视频地址：[深入浅出Pytorch](https://www.bilibili.com/video/BV1e341127Lt?p=1)

# DataWhale“深入浅出PyTorch”第五章——PyTorch模型定义的方式
@[toc]


模型在深度学习中具有重要的作用，特定的模型能更好地完成解决特定的问题。

 - **CNN**：解决图像、视频处理
 - **RNN/LSTM**：解决序列数据处理
 - **GNN**：在图模型上发挥重要作用


## 1.1 前置知识
- Module 类是 torch.nn 模块里提供的一个模型构造类 (nn.Module)，是所有神经网络模块的基类，我们可以继承它来定义我们想要的模型；
- PyTorch模型定义包括两个主要部分：各个部分的初始化（\__init__）；数据流向定义（forward）

基于nn.Module，pytorch模型的定义方式有如下三种：**Sequential，ModuleList**和**ModuleDict**

## 1.2 Sequential(顺序的)
**对应模块为nn.Sequential()**
该模型的前向计算就是将这些实例按添加的顺序逐一计算，它可以接收一个子模块的有序字典(OrderedDict) 或者一系列子模块作为参数来逐一添加 Module 的实例。
下面结合Sequential和定义方式来理解：

```python

class MySequential(nn.Module):
    from collections import OrderedDict
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict): # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些Module
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
    def forward(self, input):
        # self._modules返回一个 OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            input = module(input)
        return input

```
Sequential定义模型时只要将模型的层按序排列起来即可，根据层名不同有**直接排列**和**有序字典OrderedDict**两种方式。

- 直接排列
![在这里插入图片描述](https://img-blog.csdnimg.cn/005924ab35974c8f999153c4e5e1ca4b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)


- 有序字典OrderedDict
![在这里插入图片描述](https://img-blog.csdnimg.cn/62f4c4d99dc641668fe90abe524fab23.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
**总结**
Sequential定义模型
- 好处：简单易读且因为顺序已经订好了则不需要再写forward
- 坏处：丧失灵活性，例如不适合完成在模型中间加入一个外部输入

## 1.3 ModuleList
对应模块为nn.ModuleList()。
ModuleList 接收一个子模块（或层，需属于nn.Module类）的列表作为输入，然后也可以**类似List**那样进行行append和extend操作。同时，子模块或层的权重也会自动添加到网络中来。

ModuleList类的定义（[PyTorch文档中的ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html)）：


```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, l in enumerate(self.linears):
            x = self.linears[i // 2](x) + l(x)
        return x
```
实例化ModuleList并测试部分功能

```python
    net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
    net.append(nn.Linear(256, 10))  # # 类似List的append操作
    net.append(nn.Linear(256, 8))  

    net.extend((nn.Linear(256,7),nn.ReLU()))#extend操作,在后面追加多个值
    print(net[-1])  # 类似List的索引访问
    print(net)
```
上面代码输出结果：

```bash

ReLU()
ModuleList(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
  (3): Linear(in_features=256, out_features=8, bias=True)
  (4): Linear(in_features=256, out_features=7, bias=True)
  (5): ReLU()
)
```
**注意**
nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起。**ModuleList中元素的先后顺序并不代表其在网络中的真实位置顺序**，需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义。

## 1.4 ModuleDict
对应模块为nn.ModuleDict()。
ModuleDict和ModuleList的作用类似，只是ModuleDict能够更方便地为神经网络的层添加名称。

ModuleDict类的定义：([PyTorch文档中的ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html))

```python
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
```
实例化ModuleDict类

```python
    net = nn.ModuleDict({
        'linear': nn.Linear(784, 256),
        'act': nn.ReLU(),
    })
    net['output'] = nn.Linear(256, 10)  # 添加
    net['test'] = nn.Linear(256,11)
    print(net['linear'])  # 访问
    print(net.output)
    print(net['test'])
    print(net)
```
输出结果

```bash
Linear(in_features=784, out_features=256, bias=True)
Linear(in_features=256, out_features=10, bias=True)
Linear(in_features=256, out_features=11, bias=True)
ModuleDict(
  (linear): Linear(in_features=784, out_features=256, bias=True)
  (act): ReLU()
  (output): Linear(in_features=256, out_features=10, bias=True)
  (test): Linear(in_features=256, out_features=11, bias=True)
)
```

## 1.5 三种方法的比较与使用场景
1. Sequential适用于快速验证结果，因为已经明确了要用哪些层，直接写一下就好了，不需要同时写\__init__和forward；

2. ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；

3. 当我们需要之前层的信息的时候，比如 ResNets 中的 残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便。


## 2.1 利用模型块快速搭建复杂网络

> 上一节给出的示例都是用torch.nn中的层来完成的，这种定义方式易于理解但在实际场景中却很少被使用。当模型的深度非常大时候，使用Sequential定义模型结构需要向其中添加几百行代码，使用起来不甚方便。
> 所以对于大部分模型结构（比如ResNet、DenseNet等），虽然模型有很多层，
> 但是其中有很多重复出现的结构。考虑到每一层有其输入和输出，若干层串联成的”模块“也有其输入和输出，如果我们能将这些重复出现的层定义为一个”模块“，每次只需要向网络中添加对应的模块来构建模型，这样将会极大便利模型构建的过程。

下面将以经典的医学影响分割模型U-Net为例，介绍如何构建模型块以及如何利用模型块快速搭建复杂模型。

U-Net模型结构如下图所示，通过残差连接结构解决了模型学习中的退化问题，使得神经网络的深度能够不断扩展。
![在这里插入图片描述](https://img-blog.csdnimg.cn/5f7859d824b74cf4adf552a4550cd3f5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)

## 2.2 U-Net模型块分析

**特点**： 良好的对称性，模型从上到下分为若干层，每层由左侧和右侧两个模型块组成，每侧的模型块与其上下模型块之间有连接；同时位于同一层左右两侧的模型块之间也有连接，称为“Skip-connection”。此外还有输入和输出处理等其他组成部分。因为模型的形状类似与字母"U"，因此被命名为"U-Net".

U-Net的模型块的主要组成部分：
1）每个子块内部的两次卷积（Double Convolution，*图中蓝色箭头*）

2）左侧模型块之间的下采样连接，通过Max pooling来实现（*图中红色箭头*）

3）右侧模型块之间的上采样连接（Up sampling，*图中绿色箭头*）

4）输出层的处理（最后一个蓝绿色j箭头，1*1卷积核）

除模型块外，还有模型块之间的横向连接，输入和U-Net底部的连接等计算，这些单独的操作可以通过forward函数来实现。

## 2.3 U-Net模型块实现
具体代码太多就不贴了，自己也没弄得很懂，具体可看链接：[5.2.3部分](https://github.com/datawhalechina/thorough-pytorch/blob/main/%E7%AC%AC%E4%BA%94%E7%AB%A0%20PyTorch%E6%A8%A1%E5%9E%8B%E5%AE%9A%E4%B9%89/5.2%20%E5%88%A9%E7%94%A8%E6%A8%A1%E5%9E%8B%E5%9D%97%E5%BF%AB%E9%80%9F%E6%90%AD%E5%BB%BA%E5%A4%8D%E6%9D%82%E7%BD%91%E7%BB%9C.md)

## 3.1 修改模型层
以pytorch官方视觉库torchvision预定义好的模型ResNet50为例，探索如何修改模型的某一层或者某几层。我们先看看模型的定义是怎样的：

```python
import torchvision.models as models
net = models.resnet50()
print(net)
```
最后的输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/0584390646eb430d830209eaaad247a8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
这里模型结构是为了适配ImageNet预训练的权重，因此最后全连接层（fc）的输出节点数是1000。

假设我们要用这个resnet模型去做一个10分类的问题，就应该修改模型的fc层，将其输出节点数替换为10。另外，我们觉得一层全连接层可能太少了，想再加一层。可以做如下修改：

```python
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                          ('relu1', nn.ReLU()), 
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(128, 10)),
                          ('output', nn.Softmax(dim=1))
                          ]))
    
net.fc = classifier
print(net)
```
最后部分的输出结果如下:
![在这里插入图片描述](https://img-blog.csdnimg.cn/01a56fd5936d4461a14b7ce0e7cc396c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)将模型（net）最后名称为“fc”的层替换成了我们自己定义的名称为“classifier”的结构。

## 3.2 添加外部输入
有时候在模型训练中，除了已有模型的输入之外，还需要输入额外的信息。比如在CNN网络中，我们除了输入图像，还需要同时输入图像对应的其他信息，这时候就需要在已有的CNN网络中添加额外的输入变量。

以torchvision的resnet50模型为基础，任务还是10分类任务。不同点在于，我们希望利用已有的模型结构，在倒数第二层增加一个额外的输入变量add_variable来辅助预测。具体实现如下：


```python
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
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)),1)
        x = self.fc_add(x)
        x = self.output(x)
        return x

net = models.resnet50()
model = Model(net)
print(model)
```
最后的输出结果如下:
![在这里插入图片描述](https://img-blog.csdnimg.cn/ee164e1753a545cd9051eac76183a126.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
这里的实现要点是通过torch.cat实现了tensor的拼接。torchvision中的resnet50输出是一个1000维的tensor，我们通过修改forward函数（配套定义一些层），先将2048维的tensor通过激活函数层和dropout层，再和外部输入变量"add_variable"拼接，之后再通过全连接层映射到指定的输出维度10,也即上图的fc_add处。

## 3.3 添加额外输出
有时候在模型训练中，除了模型最后的输出外，我们需要输出模型某一中间层的结果，以施加额外的监督，获得更好的中间层结果。基本的思路是**修改模型定义中forward函数的return变量**。

```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x1000 = self.net(x)
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        return x10, x1000
 
import torchvision.models as models
net = models.resnet50()
model = Model(net)
print(model)
```
输出结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/911b2054285442e58a46e893843cfc9a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
## 4.1 PyTorch模型保存与读取
PyTorch存储模型主要采用pkl，pt，pth三种格式。

## 4.2 模型存储内容
一个PyTorch模型主要包含两个部分：模型结构和权重。其中模型是继承nn.Module的类，权重的数据结构是一个字典（key是层名，value是权重向量）。存储也由此分为两种形式：存储整个模型（包括结构和权重），和只存储模型权重。

```python
from torchvision import models
model = models.resnet152(pretrained=True)

# 保存整个模型
torch.save(model, save_dir)
# 保存模型权重
torch.save(model.state_dict, save_dir)
```

对于PyTorch而言，pt, pth和pkl**三种数据格式均支持模型权重和整个模型的存储**，因此使用上没有差别。

## 4.3 单卡和多卡模型存储与加载
详细看链接：[5.4.3~5.4.4部分](https://github.com/datawhalechina/thorough-pytorch/blob/main/%E7%AC%AC%E4%BA%94%E7%AB%A0%20PyTorch%E6%A8%A1%E5%9E%8B%E5%AE%9A%E4%B9%89/5.4%20PyTorh%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E8%AF%BB%E5%8F%96.md)
