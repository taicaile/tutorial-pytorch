import torch
import torch.optim as optim

torch.manual_seed(1)
'''
学习 optimizer 模块
'''

weight = torch.randn((2,2), requires_grad=True)
weight.grad = torch.ones((2,2))

optimizer = optim.SGD([weight], lr=1)

# ---- step ----
print("---- step ----")
flag = 0
# flag = 1
if flag:
    print(f"weight before step:{weight.data}")
    optimizer.step() # 修改lr为0.1查看结果
    print(f"weight after step:{weight.data}")

# ---- zero_grad ----
print("---- zero_grad ----")
flag = 0
# flag = 1
if flag:
    print(f"grad before zero_grad():{weight.grad}")
    optimizer.zero_grad()
    print(f"grad after zero_grad():{weight.grad}")

# ---- add_param_group ----

flag = 0
# flag = 1
if flag:
    print(f"optimizer.param_groups is \n{optimizer.param_groups}")
    w2 = torch.randn((3,3), requires_grad=True)
    optimizer.add_param_group({"params":w2, 'lr':0.0001})
    print(f"optimizer.param_groups is \n{optimizer.param_groups}")

# ---- state_dict ----
print("---- state_dict ----")
flag = 0
# flag = 1
if flag:
    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    opt_state_dict = optimizer.state_dict()

    print("state_dict before step:\n", opt_state_dict)

    for i in range(10):
        optimizer.step()

    print("state_dict after step:\n", optimizer.state_dict())

    torch.save(optimizer.state_dict(), "optimizer_state_dict.pkl")

# ---- load state_dict ----
print("---- load state_dict ----")
# flag = 0
flag = 1
if flag:
    
    optimizer = optim.SGD([weight], lr=0.1, momentum=0.9)
    state_dict = torch.load("optimizer_state_dict.pkl")

    print(f"state_dict before load state:\n{optimizer.state_dict()}\n")
    optimizer.load_state_dict(state_dict)
    print(f"state_dict after load state:\n{optimizer.state_dict()}")
