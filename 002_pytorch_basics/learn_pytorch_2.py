import math
import torch
import torch.nn as nn
import torch.optim as optim

x = torch.linspace(-math.pi, math.pi, 2000)
target = torch.sin(x)

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.c = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.d = nn.Parameter(torch.randn(1, requires_grad=True))
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True))
        self.d = nn.Parameter(torch.randn(1, requires_grad=True))
    def forward(self,x):
        x = self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3
        return x

model = Model()
optimizer = optim.SGD(model.parameters(), lr=0.000001)
riterion = torch.nn.MSELoss(reduction='sum')

for t in range(2000):
    y = model(x)
    loss = riterion(target,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(t, loss.item())

