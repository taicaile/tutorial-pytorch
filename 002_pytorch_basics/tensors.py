# -*- coding: utf-8 -*-
import torch
import numpy as np

# tensor initialization methods
# direct from data
data = [[1,2],[3,4]]
# x_data = torch.Tensor(data)
x_data = torch.tensor(data)
print(type(x_data))
print(x_data)

# from numpy array
# numpy 和 tensor 变量共享内容，修改一个，另一个也会修改
np_data = np.array(data)
x_data = torch.from_numpy(np_data)
print(type(x_data))
print(x_data)

# from another tensor with same shape, datatype
x_ones = torch.ones_like(x_data)
print(type(x_ones))
print(x_ones)

x_rand = torch.rand_like(x_data, dtype=torch.float) # override datatype
print(type(x_rand))
print(x_rand)

# with random or constant values
shape = (2,3)
x_rand = torch.rand(shape)
print(x_rand)

x_ones = torch.ones(shape)
print(x_ones)

x_full = torch.full(shape, fill_value=23)
print(x_full)

x_zeros = torch.zeros(shape)
print(x_zeros)

# tensor attribute
tensor = torch.rand(2,3)
print(tensor.shape)
print(tensor.device)
print(tensor.dtype)
print(tensor)

# tensor operation
# GPU can be changed in Colab
tensor = torch.rand(3,4)
print(tensor.device)
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
  print("move tensor to GPU memory")
print(tensor.device)

# standard numpy-like slicing and indexing
tensor = torch.ones(3,4)
tensor[1,:]=0
print(tensor)

# joining tensors, no new axis
print(torch.cat((tensor,tensor,tensor), dim=1))

# multiplying tensors
tensor = torch.ones(3,3)
tensor[:,1]=0
print(tensor)
print(tensor*tensor)
print(tensor.mul(tensor))

# matrix multiplication
print(tensor.matmul(tensor))

# In-place operations Operations that have a _ suffix are in-place. For example: x.copy_(y), x.t_(), will change x.
tensor = torch.rand(3,2)
print(tensor)
tensor.add_(5)
print(tensor)

# bridge with numpy
# Tensors on the CPU and numpy can share their underlying memory, changing one will change other.
tensor = torch.ones(3,3)
print(tensor.device)
# tensor to numpy
np_data = tensor.numpy()
# change numpy reflects in the tensor
np_data[1] = 5
print(tensor)

# 
x_steps = torch.arange(0, 100, 2)
print(x_steps)

x_lin = torch.linspace(0,100,5)
print(x_lin)

# 根据概率分布创建 tensor
x_nor = torch.normal(0., 1., size=(4,))
print(x_nor)

# 标准正太分布
print(torch.randn(12))

# 均匀分布
print(torch.rand(12))
print(torch.randint(0,10, size=(4,)))

# 张量拼接， cat 和 
t = torch.ones((2,3))
t_0 = torch.cat((t,t),dim=0)
t_1 = torch.cat((t,t),dim=1)
print(t_0.shape)
print(t_1.shape)

# stack
t_0 = torch.stack((t,t),dim=2) # 创建新的维度进行拼接
print(t_0.shape)

#张量切分 torch.chunk()
a = torch.ones((2,5))
list_of_tensor = torch.chunk(a, dim=1, chunks=2)
for t in list_of_tensor:
  print(t)
print()
# split
a = torch.ones((2,5))
# split_size_or_sections=[2,1,2]
list_of_tensor = torch.split(a, split_size_or_sections=2, dim=1)
for t in list_of_tensor:
  print(t)
print()
# index 
t = torch.randn((3,3))
idx = torch.tensor([0,2], dtype=torch.long)
t_select = torch.index_select(t,index=idx,dim=0)
print(t_select)
# mask index, 对mask中为true的数据进行索引
# torch.masked_select() , 返回一维张量
t = torch.randint(0,9,(3,3))
mask = t.ge(5) # 大于等于5的数值返回True
t_select = torch.masked_select(t, mask)
print(t_select)
print()
# 张量变换
# torch.reshape(), 当张量在内存中连续时，新张量与input共享数据内存。
t = torch.randint(0,9,(8,))
t_r = torch.reshape(t,shape=(2,4))
print(t)
print(t_r)
print()

# 张量维度变换
t = torch.rand((2,3,4))
t1=torch.transpose(t, dim0=1,dim1=2)
print(t.shape)
print(t1.shape)
print()

# 张量变换
# torch.sequeeze, 移除维度为1的dimention
t = torch.rand((1,2,3))
t1 = torch.squeeze(t)
print(t.shape)
print(t1.shape)
# torch.unsqueeze(), 拓展维度，
print()

# 张量的数学运算

# torch.add(), input + alpha*other
t0 = torch.randn((3,3))
t1 = torch.full_like(t0, fill_value=2)
t_add = torch.add(t0,10,t1)
print(t0)
print(t1)
print(t_add)
print()

# 构建线性回归模型
x = torch.rand(20,1) * 10
y = 2*x + 5 + torch.randn(20,1)

# 构建线性回归参数
w = torch.randn((1,), requires_grad=True)
b = torch.zeros((1,), requires_grad=True)
lr = 0.01

for iteration in range(1000):
  # 向前传播
  wx = torch.mul(w, x)
  y_pred = torch.add(wx,b)

  # 计算loss
  loss = (0.5*(y_pred-y)**2).mean()

  # 反向传播
  loss.backward()
  
  # 更新参数
  b.data.sub_(lr*b.grad)
  w.data.sub_(lr*w.grad)
  w.grad.data.zero_()
  b.grad.data.zero_()

  if iteration%100==0:
    import matplotlib.pyplot as plt
    plt.scatter(x.data.numpy(),y.data.numpy())
    plt.plot(x.data.numpy(),y_pred.data.numpy(),'r-', lw=5)
    plt.text(2,20,f"loss:{loss.item()}",fontdict={'size':20, 'color':'red'})
    plt.xlim(1,5,10)
    plt.ylim(0,30)
    plt.title(f"iteration:{iteration}\n w:{w.data.numpy()}, b:{b.data.numpy()}")
    plt.pause(0.5)
print(w.item(), b.item())