import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

x = torch.linspace(-math.pi, math.pi, 2000)
target = torch.sin(x)

a = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)
c = torch.randn(1, requires_grad=True)
d = torch.randn(1, requires_grad=True)

optimizer = optim.SGD([a,b,c,d], lr=0.000001)
for t in range(2000):
    y = a + b*x + c*x**2 + d*x** 3
    loss = ((y-target)**2).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(t, loss.item())
