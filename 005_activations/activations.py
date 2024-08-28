import torch
import torch.nn as nn
import matplotlib.pyplot as plt

acts = [nn.ReLU(), nn.Tanh(), nn.Sigmoid(), nn.SiLU(), nn.Mish()]
x = torch.linspace(-10,10,1000)
ys = [act(x) for act in acts]
n = len(acts)

plt.figure()
axes = plt.gca()
# plt.axis('equal')
# axes.set_ylim([-15,15])
for i in range(n):
    plt.plot(x.numpy(), ys[i].numpy(), linewidth=2, label=str(acts[i]))
plt.legend()
# plt.show()
plt.savefig('acts.png', dpi=300)