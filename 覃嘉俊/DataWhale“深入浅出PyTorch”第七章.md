教程前四章的打卡记录：[深入迁出pytorch专栏](https://blog.csdn.net/soraca/category_11409115.html)
该教程的GitHub地址：[深入浅出PyTorch
](https://github.com/datawhalechina/thorough-pytorch)哔哩哔哩视频地址：[深入浅出Pytorch](https://www.bilibili.com/video/BV1e341127Lt?p=1)

# DataWhale“深入浅出PyTorch”第七章—— PyTorch可视化
*ps:本次主要都是解决	BUG了*
@[toc]
## 1可视化网络结构
随着深度神经网络做的的发展，网络的结构越来越复杂，我们也很难确定每一层的输入结构，输出结构以及参数等信息，这样导致我们很难在短时间内完成debug。因此掌握一个可以用来可视化网络结构的工具是十分有必要的。

深度学习库**Keras**中可以调用一个叫做`model.summary()`的API来很方便地实现，调用后就会显示我们的模型参数，输入大小，输出大小，模型的整体参数等。

而在**pytorch**中，人们开发了torchinfo工具包 ( torchinfo是由torchsummary和torchsummaryX重构出的库, torchsummary和torchsummaryX已经许久没更新了) 

### 1.1 使用print函数打印模型基础信息

```python
import torchvision.models as models
model = models.resnet18()
print(model)#打印Resnet18的模型结构
```
部分输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/1cce9c41a8404d5bbe26cd433caaa714.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)

单纯的print(model)，只能得出基础构件的信息，既不能显示出每一层的shape，也不能显示对应参数量的大小，为了解决这些问题，我们就需要使用`torchinfo`。

### 1.2 使用torchinfo可视化网络结构
**torch的安装**

- 方法一:pip install torchinfo 

- 方法二：conda install -c conda-forge torchinfo

**torchinfo的使用**
只需要使用`torchinfo.summary()`就行了，必需的参数分别是model，input_size[batch_size,channel,h,w]，更多参数可以参考[documentation](https://github.com/TylerYep/torchinfo#documentation)
```python
import torchvision.models as models
from torchinfo import summary
resnet18 = models.resnet18() # 实例化模型
summary(model, (1, 3, 224, 224)) # 1：batch_size 3:图片的通道数 224: 图片的高宽
```
**torchinfo的结构化输出结果**

![在这里插入图片描述](https://img-blog.csdnimg.cn/af62d784122942ec9d0083aa38c7bf80.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)

![在这里插入图片描述](https://img-blog.csdnimg.cn/7dcb9416cfa74405a5a40e07a49d60ce.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
可以看到torchinfo提供了更加详细的信息，包括模块信息（每一层的类型、输出shape和参数量）、模型整体的参数量、模型大小、一次前向或者反向传播需要的内存大小等。


## 2 CNN可视化
卷积神经网络（CNN）是深度学习中非常重要的模型结构，它广泛地用于图像处理，极大地提升了模型表现，推动了计算机视觉的发展和进步。但C**NN是一个“**黑盒模型**”，人们并不知道CNN是如何获得较好表现的，由此带来了深度学习的可解释性问题**。如果能理解CNN工作的方式，人们不仅能够解释所获得的结果，提升模型的鲁棒性，而且还能有针对性地改进CNN的结构以获得进一步的效果提升。

理解CNN的重要一步是可视化，包括可视化特征是如何提取的、提取到的特征的形式以及模型在输入数据上的关注点等。

### 2.1 CNN卷积核可视化
卷积核在CNN中负责提取特征，可视化卷积核能够帮助人们理解CNN各个层在提取什么样的特征，进而理解模型的工作原理。例如在Zeiler和Fergus 2013年的[paper](https://arxiv.org/pdf/1311.2901.pdf)中就研究了CNN各个层的卷积核的不同，他们发现靠近输入的层提取的特征是相对简单的结构，而靠近输出的层提取的特征就和图中的实体形状相近了(**即越深层学习到的越深层细节的特征**)，如下图所示：
![请添加图片描述](https://img-blog.csdnimg.cn/62f4990c7bab4a77b5064545005f950b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
![请添加图片描述](https://img-blog.csdnimg.cn/f05281a708ad403ca7d4e040d88bb9db.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
![请添加图片描述](https://img-blog.csdnimg.cn/69be8bed5d1f453d8c0714591c85ced7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_12,color_FFFFFF,t_70,g_se,x_16)
在PyTorch中可视化卷积核也非常方便，核心在于特定层的卷积核即特定层的模型权重，可视化卷积核就等价于可视化对应的权重矩阵。下面给出在PyTorch中可视化卷积核的实现方案，以torchvision自带的VGG11模型为例。

首先加载模型，并确定模型的层信息：

```python
import torch
from torchvision.models import vgg11

model = vgg11(pretrained=True)
print(dict(model.features.named_children()))
```



```bash
{'0': Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '1': ReLU(inplace=True),
 '2': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '3': Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '4': ReLU(inplace=True),
 '5': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '6': Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '7': ReLU(inplace=True),
 '8': Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '9': ReLU(inplace=True),
 '10': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '11': Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '12': ReLU(inplace=True),
 '13': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '14': ReLU(inplace=True),
 '15': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
 '16': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '17': ReLU(inplace=True),
 '18': Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
 '19': ReLU(inplace=True),
 '20': MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)}
```
卷积核对应的应为卷积层（Conv2d），这里以第“3”层为例，可视化对应的参数：
**报figure.max_open_warning错可加句代码阻止此警告**
```python
conv1 = dict(model.features.named_children())['3']
kernel_set = conv1.weight.detach()
num = len(conv1.weight.detach())
print(kernel_set.shape)
#阻止发出“figure.max_open_warning”警告
plt.rcParams.update({'figure.max_open_warning': 0})
for i in range(0,num):
    i_kernel = kernel_set[i]
    plt.figure(figsize=(20, 17))
    if (len(i_kernel)) > 1:
        for idx, filer in enumerate(i_kernel):
            plt.subplot(9, 9, idx+1) 
            plt.axis('off')
            plt.imshow(filer[ :, :].detach(),cmap='bwr')
            plt.show()#如果是在pycharm跑需加上这句
```
输出结果（Google colab下）：
一堆类似下面的卷积图
![在这里插入图片描述](https://img-blog.csdnimg.cn/f08a3feadda7479db6d3c4533c08fa38.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_16,color_FFFFFF,t_70,g_se,x_16)

### 2.2 CNN特征图可视化方法
class activation map （CAM）的作用是判断哪些变量对模型来说是重要的，在CNN可视化的场景下，即判断图像中哪些像素点对预测结果是重要的。除了确定重要的像素点，人们也会对重要区域的梯度感兴趣，因此在CAM的基础上也进一步改进得到了Grad-CAM（以及诸多变种）。CAM和Grad-CAM的示例如下图所示：

**安装**

> pip install grad-cam


教程报错的一些解决方法：
- input_tensor没有，使用 preprocess_image归一化图像，将前面的算出来的rgb_img转化为tensor，
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

- preds未设定，比如ImageNet有1000类，这里可以设为200


代码如下：                                        
```python
import torch
from torchvision.models import vgg11,resnet18,resnet101,resnext101_32x8d
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

model = vgg11(pretrained=True)
img_path = './PytorchStudy/dog1.jpg'#注意图片的路径，可自行更换图片查看效果
# resize操作是为了和传入神经网络训练图片大小一致
img = Image.open(img_path).resize((224,224))
# 需要将原始图片转为np.float32格式并且在0-1之间
rgb_img = np.float32(img)/255
plt.imshow(img)
plt.show()


from pytorch_grad_cam import GradCAM,ScoreCAM,GradCAMPlusPlus,AblationCAM,XGradCAM,EigenCAM,FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# preprocess_image作用：归一化图像，并转成tensor
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
target_layers = [model.features[-1]]
# 选取合适的类激活图，但是ScoreCAM和AblationCAM需要batch_size
cam = GradCAM(model=model,target_layers=target_layers)
targets = [ClassifierOutputTarget(200)]
# 上方preds需要设定，比如ImageNet有1000类，这里可以设为200

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]
cam_img = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
print(type(cam_img))
Image.fromarray(cam_img)
plt.imshow(cam_img)
plt.show()
```

效果如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/e3e0bff67bcf46458f28bcdf2b2e1fbf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_16,color_FFFFFF,t_70,g_se,x_16#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/540b6bd656b943e3af39b3e757e3f33a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_16,color_FFFFFF,t_70,g_se,x_16#pic_center)
### 2.3 使用FlashTorch快速实现CNN可视化
**可视化梯度**

按教程的步骤来的话最后还是报错。
原因是原作者限制了只在他开发的torch和torchvision版本可以work，想体验效果可移步至：[链接](https://github.com/MisaOgura/flashtorch/issues/39)，使用Google colab体验

**可视化卷积核**

```python
import torchvision.models as models
from flashtorch.activmax import GradientAscent

model = models.vgg16(pretrained=True)
g_ascent = GradientAscent(model.features)

# specify layer and filter info
conv5_1 = model.features[24]
conv5_1_filters = [45, 271, 363, 489]

g_ascent.visualize(conv5_1, conv5_1_filters, title="VGG16: conv5_1")
```


输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/c2569ccd45ab496f9654ae12391bd999.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
## 3 使用TensorBoard可视化训练过程

*ps:教程末尾贴出的本节参考链接解决了许多问题：[博客](https://blog.csdn.net/Python_Ai_Road/article/details/107704530)*

训练过程的可视化在深度学习模型训练中扮演着重要的角色。学习的过程是一个优化的过程，我们需要找到最优的点作为训练过程的输出产物。一般来说，我们会结合训练集的损失函数和验证集的损失函数，绘制两条损失函数的曲线来确定训练的终点，找到对应的模型用于测试。
TensorBoard作为一款可视化工具除了记录训练中每个epoch的loss值，还能实时观察损失函数曲线的变化，及时捕捉模型的变化，同时也可以可视化其他内容，如输入数据（尤其是图片）、模型结构、参数分布等，这些对于我们在debug中查找问题来源非常重要（比如输入数据和我们想象的是否一致）。
### 3.1 TensorBoard安装
在已安装PyTorch的环境下使用pip安装即可：

```bash
pip install tensorboard
```

也可以使用PyTorch自带的tensorboard工具，此时不需要额外安装tensorboard。

### 3.2 TensorBoard可视化的基本逻辑

可以将TesorBoard看做一个记录员，它可以记录我们指定的数据，包括模型每一层的feature map，权重，以及训练loss等等。TensorBoard将记录下来的内容保存在一个用户指定的文件夹里，程序不断运行中TensorBoard会不断记录。记录下的内容可以通过网页的形式加以可视化。

### 3.3 TensorBoard的配置与启动

在使用TensorBoard前，我们需要先指定一个文件夹供TensorBoard保存记录下来的数据。然后调用tensorboard中的SummaryWriter作为上述“记录员”

```python
from tensorboardX import SummaryWriter

writer = SummaryWriter('./runs')
```

上面的操作实例化SummaryWritter为变量writer，并指定writer的输出目录为当前目录下的"runs"目录。也就是说，之后tensorboard记录下来的内容都会保存在runs。

如果使用PyTorch自带的tensorboard，则采用如下方式import：

```python
from torch.utils.tensorboard import SummaryWriter
```

可以手动往runs文件夹里添加数据用于可视化，或者把runs文件夹里的数据放到其他机器上可视化,只要将这个数据分享给其他人，其他人在安装了tensorboard的情况下就会看到你分享的数据。

启动tensorboard也很简单，在命令行中输入

```bash
tensorboard --logdir=/path/to/logs/ --port=xxxx
```

其中“path/to/logs/"是指定的保存tensorboard记录结果的文件路径（等价于上面的“./runs"，port是外部访问TensorBoard的端口号，可以通过访问ip:port访问tensorboard，这一操作和jupyter notebook的使用类似。**如果不是在服务器远程使用的话则不需要配置port**。**不配置的话默认端口是6006**

### 3.4 TensorBoard模型结构可视化
以Google colab环境为例：

1 安装torchkeras库

```bash
!pip install torchkeras #“!”感叹号是colab中执行命令所需 如果是cd命令用%
```
2 定义模型

```python
import torch 
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchkeras import Model,summary

class Net(nn.Module):
   def __init__(self):
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size = 3)
    self.pool = nn.MaxPool2d(kernel_size = 2,stride = 2)
    self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size = 5)
    self.adaptive_pool = nn.AdaptiveMaxPool2d((1,1))
    self.flatten = nn.Flatten()
    self.linear1 = nn.Linear(64,32)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(32,1)
    self.sigmoid = nn.Sigmoid()

   def forward(self,x):
    x = self.conv1(x)
    x = self.pool(x)
    x = self.conv2(x)
    x = self.pool(x)
    x = self.adaptive_pool(x)
    x = self.flatten(x)
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)
    y = self.sigmoid(x)
    return y

net = Net()
print(net)
summary(net,input_shape= (3,32,32))
```
输出结果：
![在这里插入图片描述](https://img-blog.csdnimg.cn/6fbf4638882f4048a269a1828495ba0e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)

3 执行三段命令

```bash
%load_ext tensorboard
#%tensorboard --logdir ./data/tensorboard
```

```bash
from tensorboard import notebook
#查看启动的tensorboard程序
notebook.list() 
```

```bash
#启动tensorboard程序
notebook.start("--logdir ./data/tensorboard")
#等价于在命令行中执行 tensorboard --logdir ./data/tensorboard
#可以在浏览器中打开 http://localhost:6006/ 查看
```

4 向模型输入

```python
writer = SummaryWriter('./data/tensorboard')
writer.add_graph(net,input_to_model = torch.rand(1,3,32,32))#input
writer.close()
notebook.start("--logdir ./data/tensorboard")#关键代码启动
```
显示结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/3c441e1a2c8e44b293e77077ceb261a3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)


双击Net后查看的网络结构图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/7b666e5754f94f0ea423ced54d307cd5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)

### 3.5 TensorBoard图像可视化
当我们做图像相关的任务时，可以方便地将所处理的图片在tensorboard中进行可视化展示。

- 对于单张图片的显示使用add_image
- 对于多张图片的显示使用add_images
- 有时需要使用torchvision.utils.make_grid将多张图片拼成一张图片后，用writer.add_image显示

这里我们使用torchvision的CIFAR10数据集为例： 
```python
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform_train = transforms.Compose(
    [transforms.ToTensor()])
transform_test = transforms.Compose(
    [transforms.ToTensor()])

train_data = datasets.CIFAR10(".", train=True, download=True, transform=transform_train)
test_data = datasets.CIFAR10(".", train=False, download=True, transform=transform_test)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

images, labels = next(iter(train_loader))
```

**仅查看一张图片**

```python
writer = SummaryWriter('./pytorch_tb')
writer.add_image('images[0]', images[0])
writer.close()
#
notebook.start("--logdir ./pytorch_tb")#等价于在命令行中执行 tensorboard --logdir ./pytorch_tb
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/8efd706795894e09a0663f573ae165e8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)

 **将多张图片拼接成一张图片，中间用黑色网格分割**

```python
# create grid of images
writer = SummaryWriter('./pytorch_tb')
img_grid = torchvision.utils.make_grid(images)
writer.add_image('image_grid', img_grid)
writer.close()
notebook.start("--logdir ./pytorch_tb")
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/d02e7cd0dda842d9a0097b21d8927085.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)

**将多张图片直接写入**

```python
writer = SummaryWriter('./pytorch_tb')
writer.add_images("images",images,global_step = 0)
writer.close()
notebook.start("--logdir ./pytorch_tb")

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/8ef9af29226a469ab7409d40eba6dc38.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
另外注意上方menu部分，刚刚只有“GRAPHS"栏对应模型的可视化，现在则多出了”IMAGES“栏对应图像的可视化。左侧的滑动按钮**可以调整图像的亮度和对比度。**

### 3.6 TensorBoard连续变量可视化

TensorBoard可以用来可视化连续变量（或时序变量）的变化过程，通过add_scalar实现：

```python
writer = SummaryWriter('./pytorch_tb')
for i in range(500):
    x = i
    y = x**2
    writer.add_scalar("x", x, i) #日志中记录x在第step i 的值
    writer.add_scalar("y", y, i) #日志中记录y在第step i 的值
writer.close()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/45fb0ee28cdb49fc9e4df0eab1bd2137.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)



如果想在同一张图中显示多个曲线，则需要分别建立存放子路径（使用SummaryWriter指定路径即可自动创建，但需要在tensorboard运行目录下），同时在add_scalar中修改曲线的标签使其一致即可：

```python
writer1 = SummaryWriter('./pytorch_tb/x')
writer2 = SummaryWriter('./pytorch_tb/y')
for i in range(500):
    x = i
    y = x*2
    writer1.add_scalar("same", x, i) #日志中记录x在第step i 的值
    writer2.add_scalar("same", y, i) #日志中记录y在第step i 的值
writer1.close()
writer2.close()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/ce1839914cd442a6aba8a71a95c18a7b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)
这部分功能非常适合损失函数的可视化，可以帮助我们更加直观地了解模型的训练情况，从而确定最佳的checkpoint。左侧的Smoothing滑动按钮可以调整曲线的平滑度，当损失函数震荡较大时，将Smoothing调大有助于观察loss的整体变化趋势。

### 3.7 TensorBoard参数分布可视化
当我们需要对参数（或向量）的变化，或者对其分布进行研究时，可以方便地用TensorBoard来进行可视化，通过add_histogram实现。下面给出一个例子：

```python
import torch
import numpy as np

# 创建正态分布的张量模拟参数矩阵
def norm(mean, std):
    t = std * torch.randn((100, 20)) + mean
    return t
 
writer = SummaryWriter('./pytorch_tb/')
for step, mean in enumerate(range(-10, 10, 1)):
    w = norm(mean, 1)
    writer.add_histogram("w", w, step)
    writer.flush()
writer.close()
```

结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/8ccb4142c16446ce9bc2fab43cd3f1c4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcXFxcXFxcWppYWp1bg==,size_20,color_FFFFFF,t_70,g_se,x_16)




### 3.8 总结
对于TensorBoard来说，它的功能是很强大的，主要的实现方案是构建一个SummaryWriter，然后通过`add_XXX()`函数来实现。

其实TensorBoard的逻辑还是很简单的，它的**基本逻辑就是文件的读写逻辑**，写入想要可视化的数据，然后TensorBoard自己会读出来。



