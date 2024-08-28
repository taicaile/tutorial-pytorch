import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim

torch.manual_seed(1)
 
n_hidden = 200
max_iter = 2000
disp_interval = 200
lr_init = 0.01

# ---- step 1, data ----

def gen_data(num_data=10, x_range=(-1,1)):
    
    w = 1.5
    train_x = torch.linspace(*x_range, num_data).unsqueeze_(1)
    train_y = w*train_x + torch.normal(0, 0.5, size=train_x.size())
    test_x = torch.linspace(*x_range, num_data).unsqueeze_(1)
    test_y = w*test_x + torch.normal(0, 0.3, size=test_x.size())

    return train_x, train_y, test_x, test_y
train_x, train_y, test_x, test_y = gen_data()

# ---- step 2, model ----

class MLP(nn.Module):
    def __init__(self, neural_num, d_prob=0.5):
        super(MLP, self).__init__()
        self.linears = nn.Sequential(
            nn.Linear(1, neural_num),
            nn.ReLU(inplace=True),

            nn.Dropout(d_prob),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=True),
            
            nn.Dropout(d_prob),
            nn.Linear(neural_num, neural_num),
            nn.ReLU(inplace=True),
            
            nn.Dropout(d_prob),
            nn.Linear(neural_num, 1),
        )

    def forward(self, x):
        return self.linears(x)

net_prob_0 = MLP(neural_num=n_hidden, d_prob=0)
net_prob_0_5 = MLP(neural_num=n_hidden, d_prob=0.5)

# ---- optimizition ----
optim_prob_0 = optim.SGD(net_prob_0.parameters(), lr=lr_init, momentum=0.9)
optim_prob_05 = optim.SGD(net_prob_0.parameters(), lr=lr_init, momentum=0.9)

# ---- loss ----

loss_func = nn.MSELoss()

# ---- train ----

writer = SummaryWriter()

for epoch in range(max_iter):

    # forward
    pred_normal, pred_decay = net_prob_0(train_x), net_prob_0_5(train_x)
    loss_normal, loss_decay = loss_func(pred_normal, train_y), loss_func(pred_decay, train_y)

    optim_prob_0.zero_grad()
    optim_prob_05.zero_grad()

    loss_normal.backward()
    loss_decay.backward()

    optim_prob_0.step()
    optim_prob_05.step()

    if (epoch+1)%disp_interval == 0:
        # 可视化
        net_prob_0.eval()
        net_prob_0_5.eval()
        for name, layer in net_prob_0.named_parameters():
            # 记录梯度
            writer.add_histogram(name+"_grad_normal", layer.grad, epoch)
            # 记录什么?
            writer.add_histogram(name+"_data_normal", layer, epoch)

        for name, layer in net_prob_0_5.named_parameters():
            writer.add_histogram(name+"_grad_weight_decay", layer.grad, epoch)
            writer.add_histogram(name+"_data_weight_decay", layer, epoch)
        net_prob_0.train()
        net_prob_0_5.train()
# 观察 tensorboard distribution
writer.close()