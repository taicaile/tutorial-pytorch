import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(1)

# 学习率调整
# 1. StepLR，等间隔调整学习率
# lr = lr * gamma

LR = 0.1
iteration = 10
max_epoch = 200

# ---- fake data ----
LR = 0.1
weights = torch.randn((1), requires_grad=True)
target = torch.zeros((1))

optimizer = optim.SGD([weights],lr=LR, momentum=0.9)

# ---- scheduler ----
flag = False
# flag = True
if flag:
    scheduler_lr = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    lr_list, epoch_list = [],[]

    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            loss = torch.pow((weights - target), 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list,  '-r', label='LR Scheduler')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()

# ---- MultiStepLR ----
flag = False
# flag = True
if flag:
    scheduler_lr = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 120, 180], gamma=0.1)
    lr_list, epoch_list = [],[]

    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            loss = torch.pow((weights - target), 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list,  '-r', label='MultiStepLR Scheduler')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# ---- ExponentialLR ----
flag = False
# flag = True
if flag:
    scheduler_lr = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    lr_list, epoch_list = [],[]

    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            loss = torch.pow((weights - target), 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list,  '-r', label='ExponentialLR Scheduler')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()

# ---- CosineAnnealingLR ----
# cos 周期调整学习率
flag = False
# flag = True
if flag:
    scheduler_lr = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
    lr_list, epoch_list = [],[]

    for epoch in range(max_epoch):
        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)

        for i in range(iteration):
            loss = torch.pow((weights - target), 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()

    plt.plot(epoch_list, lr_list,  '-r', label='CosineAnnealingLR Scheduler')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()


# ---- CosineAnnealingLR ----
# cos 周期调整学习率
flag = False
# flag = True
if flag:
    loss_value = 0.5
    accuracy = 0.9
    
    factor = 0.1
    mode = 'min'
    patience = 10
    cooldown = 10
    min_lr = 1e-4
    verbose = True

    scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, mode=mode, patience=patience,
                                                        cooldown=cooldown, min_lr=min_lr, verbose=verbose)
    lr_list, epoch_list = [],[]

    for epoch in range(max_epoch):
        for i in range(iteration):
            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step(loss_value)

# ---- Lambda ----
flag = False
flag = True
if flag:

    lr_init = 0.1
    weights_1 = torch.randn((6,3,5,5))
    weights_2 = torch.ones((5,5))
    optimizer = optim.SGD([{'params':[weights_1]},
                           {'params':[weights_2]}], lr=lr_init)

    lambda1 = lambda epoch: 0.1**(epoch//20)
    lambda2 = lambda epoch: 0.95 ** epoch 

    scheduler_lr = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
    lr_list, epoch_list = [],[]

    for epoch in range(max_epoch):
        for i in range(iteration):
            optimizer.step()
            optimizer.zero_grad()

        scheduler_lr.step()
        lr_list.append(scheduler_lr.get_lr())
        epoch_list.append(epoch)
        print(f"{epoch}:{scheduler_lr.get_lr()}")

    plt.plot(epoch_list, lr_list,  '-r', label='Lambda Scheduler')
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.show()