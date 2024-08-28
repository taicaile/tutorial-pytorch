import torch
import matplotlib.pyplot as plt

torch.manual_seed(1)

def func(x_t):
    """
    y = (2x)^2 = 4*x^2   dy/dx=8x
    """
    return torch.pow(2*x_t, 2)

# init
x = torch.tensor([2.], requires_grad=True)

# ---- plot data ----
flag = 0
# flag = 1
if flag:
    x_t = torch.linspace(-3,3,100)
    y = func(x_t)
    
    plt.plot(x_t.numpy(), y.numpy(), label="y=4*x^2")
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    

# ---- 
# flag = False
flag = True
if flag:
    x_t = torch.linspace(-3,3,100)
    fun_y = func(x_t)
    iter_rec, loss_rec, x_rec = list(), list(), list()
    lr = 0.2
    max_iter = 40
    for i in range(max_iter):
        y = func(x)
        y.backward()
        print(f"Iter:{i}, x:{x.detach().numpy()[0]:8}, x.grad:{x.grad.detach().numpy()[0]:8}, loss:{y.item():10}")

        x_rec.append(x.item())

        x.data.sub_(lr*x.grad)
        x.grad.zero_()

        iter_rec.append(i)
        loss_rec.append(y.item())
    plt.subplot(121).plot(iter_rec, loss_rec, '-ro')
    plt.xlabel("x")
    plt.ylabel("loss")

    ax = plt.subplot(122)
    ax.plot(x_t.numpy(), fun_y.numpy(), label="y=4*x^2")
    ax.plot(x_rec, [func(torch.tensor(x)).item() for x in x_rec], '-ro')
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()