# calculate gradient of x and w for function "y=(w+x)(w+1)"

import torch

w = torch.tensor([1.], requires_grad=True)
x = torch.tensor([2.], requires_grad=True)

a = torch.add(w,x)
b = torch.add(w,1)
y = torch.mul(a,b)

# 求出y对所有叶子节点的梯度
# 实际调用 torch.autograd.backward 方法
y.backward()
print(w.grad)

#-------------------------------------------------------
# torch.autograd.grad 也是求导数，可以指定输入和输出
#-------------------------------------------------------

x = torch.tensor([3.], requires_grad=True)
y = torch.pow(x,2) # y = x**2, y' = 2x

grad1 = torch.autograd.grad(y,x,create_graph=True)
print(grad1)

# 叶子节点不能执行 in-place 操作
