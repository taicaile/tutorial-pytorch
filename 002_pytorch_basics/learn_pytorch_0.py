import math
import torch
import numpy as np

x = torch.linspace(-math.pi, math.pi, 2000)
target = torch.sin(x)

a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
c = torch.randn(1, requires_grad=True)
d = torch.randn(1, requires_grad=True)

lr = 1e-6
for t in range(2000):
    y = a + b*x + c*x**2 + d*x**3
    loss = ((y-target)**2).sum()
    loss.backward()
    a.data.sub_(a.grad*lr)
    b.data.sub_(b.grad*lr)
    c.data.sub_(c.grad*lr)
    d.data.sub_(d.grad*lr)
    a.grad.zero_()
    b.grad.zero_()
    c.grad.zero_()
    d.grad.zero_()
    print(t, loss.item())

