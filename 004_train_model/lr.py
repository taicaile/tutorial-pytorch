import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(10)

# build a logistic regression

# generate data
bias = 50
mean_value = 1.7 # 均值
sample_nums = 100 

n_data = torch.ones(sample_nums, 2) # n_data.shape = 100,2, 两个属性

x0 = torch.normal(mean_value*n_data, 1) + bias
y0 = torch.zeros(sample_nums) # 标签为0

x1 = torch.normal(-mean_value * n_data, 1) + bias
y1 = torch.ones(sample_nums) # 标签为1

train_x = torch.cat((x0,x1),0)
train_y = torch.cat((y0,y1),0)

# print(train_x.shape) # 200,2
# print(train_y.shape) # 200
# build model

class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.features = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.features(x)
        return self.sigmoid(x)

lr_net = LR() # 实例化逻辑回归模型

# loss function
loss_fn = nn.BCELoss()

# optimizer

lr = 0.01
optimizer = optim.SGD(lr_net.parameters(), lr=lr, momentum=0.9)

# 

for i in range(2000):

    y_pred = lr_net(train_x)
    loss = loss_fn(y_pred.squeeze(), train_y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i%200==199:
        import matplotlib.pyplot as plt
        mask = y_pred.ge(0.5).float().squeeze()
        correct = (mask==train_y).sum()
        acc = correct.item()/train_y.size(0)

        plt.scatter(x0.data.numpy()[:,0], x0.data.numpy()[:,1], c='r', label='class_0')
        plt.scatter(x1.data.numpy()[:,0], x1.data.numpy()[:,1], c='b', label='class_1')
        
        w0, w1 = lr_net.features.weight[0]
        w0, w1 = float(w0.item()), float(w1.item())
        plot_b = float(lr_net.features.bias[0].item())
        plot_x = np.arange(-6,6,0.1)
        plot_y = (-w0 * plot_x - plot_b)/w1

        plt.plot(plot_x, plot_y)
        plt.text(-5,5, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color':'red'})

        plt.title(f"iteration: {i:.2f}\n w0:{w0:.2f}, w1:{w1:.2f} accuracy:{acc:.2f}")
        plt.legend()
        plt.show(block=False)
input('press <ENTER> to continue')
