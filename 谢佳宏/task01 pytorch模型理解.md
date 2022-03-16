# **1.pytorch的模型定义**



pytorch分为三种模型定义方式：

- Sequential
- ModuleList
- ModuleDict

#### **Sequential**

```python
# 一般使用直接排列
import torch.nn as nn
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
print(net)
"""
output:
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
"""

# 使用OrderedDict
import collections
import torch.nn as nn
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
print(net2)
"""
output:
Sequential(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=256, out_features=10, bias=True)
)
"""
```



#### **ModuleList**和ModuleDict

```python
# ModuleList
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问
print(net)
"""
output:
Linear(in_features=256, out_features=10, bias=True)
ModuleList(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
"""

# ModuleDict
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)

"""
output:
Linear(in_features=784, out_features=256, bias=True)
Linear(in_features=256, out_features=10, bias=True)
ModuleDict(
  (act): ReLU()
  (linear): Linear(in_features=784, out_features=256, bias=True)
  (output): Linear(in_features=256, out_features=10, bias=True)
)
"""
```

可以看出ModuleList和moduleDict作用类似 ，区别在于给层命名了。但要注意的是他们并**没有定义一个网络，它只是将不同的模块储存在一起**。其中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义。

其实个人认为还是按照__init__()和__forward__()的方式构建模型清晰灵活。



# **2. 利用模型模块搭建复杂网络**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class submodel_1(nn.Module):
  def __init__(self, param_list):
    super(submodel_1, self).__init__()
    self.params = param_list
    self.layer_list = [Linear(), Linear()]
  def forward(self, x):
    for layer in self.layer_list:
      x = layer(x)
    return x
  
 class submodel_2(nn.Module):
  def __init__(self, param_list):
    super(submodel_2, self).__init__()
    self.params = param_list
    self.layer_list = [Linear(), Linear()]
  def forward(self, x):
    for layer in self.layer_list:
      x = layer(x)
    return x
  
  
  class main_model(nn.Module):
    def __init__(self, param_list):
      super(main_model, self).__init__()
      self.param = param_list
     	self.sub_1 = submodel_1
      self.sub_2 = submodel_2
     	
    def forward(x):
      out_1 = self.sub_1(x)
      out_2 = self.sub_2(x)
      out = opt(out_1, out_2)
      return out
```

这里主要是利用继承nn.Module这个类构建一些模型内的一些模块，然后在最终的main_model里面复用定义好的模块来进行前向图的构建。**若是不想使梯度回传到模块中，记得使用with torch.no_grad()方法将不需要回传梯度的模块放到包裹起来**。

# **3. 修改模型**

往往我们有时候会需要对一些开源的模型进行定制化的修改，增加输入或者利用模型得到中间输出。在pytorch中主要是在forward部分进行修改得到

```python
# 在这里多传入一个参数达到增加输入的目的
def forward(x, add_on_variable):
  out_1 = self.sub_1(x)
  out_2 = self.sub_2(x)
  out = opt(out_1, out_2, add_on_variable)
  return out

# 在return处多返回需要的值，达到输出中间值的目的 
def forward(x, add_on_variable):
  out_1 = self.sub_1(x)
  out_2 = self.sub_2(x)
  out = opt(out_1, out_2, add_on_variable)
  return out, out_1
```

