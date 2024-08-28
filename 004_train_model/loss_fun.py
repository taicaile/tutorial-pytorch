import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

# ---- cross entropy loss ----
flag = True
flag = False

if flag:

    inputs = torch.tensor([[1,2],[1,3],[1,3]],dtype=torch.float)
    target = torch.tensor([0,1,1], dtype=torch.long)

    # crossentropy loss
    loss_f_none = nn.CrossEntropyLoss(weight=None, reduction='none')
    loss_f_sum = nn.CrossEntropyLoss(weight=None, reduction='sum')
    loss_f_mean = nn.CrossEntropyLoss(weight=None, reduction='mean')

    # forward
    loss_none = loss_f_none(inputs, target)
    loss_sum = loss_f_sum(inputs, target)
    loss_mean = loss_f_mean(inputs, target)

    # view
    print("cross entropy loss:\n", loss_none, loss_sum, loss_mean)

    # 如果指定 weight 为 [1,2]，则类别0的权值为1,类别2的权值为2, 求均值的时候，不是除样本的个数，而是除以权值累加

# ---- L1 Loss ----

# flag = True
flag = False

if flag:
    inputs = torch.ones((2,2))
    target = torch.ones((2,2)) * 3

    loss_fun = nn.L1Loss(reduction='none')
    loss = loss_fun(inputs, target)

    print(f"input:{inputs}\ntarget:{target}\nL1 Loss:{loss}")

# ---- MSE Loss ----

flag = True
# flag = False

if flag:
    inputs = torch.ones((2,2))
    target = torch.ones((2,2)) * 3

    loss_fun = nn.MSELoss(reduction='none')
    loss = loss_fun(inputs, target)

    print(f"input:{inputs}\ntarget:{target}\nL1 Loss:{loss}")

# ---- L1 Smooth Loss ----

flag = True
# flag = False

if flag:
    inputs = torch.linspace(-3,3,steps=500)
    target = torch.zeros_like(inputs)

    loss_fun = nn.SmoothL1Loss(reduction='none')
    loss_smooth = loss_fun(inputs, target)
    loss_l1 = np.abs(inputs.numpy())

    plt.plot(inputs.numpy(), loss_smooth.numpy(), label="Smooth L1 Loss")
    plt.plot(inputs.numpy(), loss_l1, label="L1 Loss")
    plt.legend()
    plt.grid()
    plt.show()