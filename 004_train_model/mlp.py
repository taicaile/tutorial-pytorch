import torch
import torch.nn as nn

import numpy as np

# ---- 未添加激活函数的MLP ---- 
class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            # 打印每一层的方差
            print(f"layer {i:02}: {x.std()}")
            if torch.isnan(x.std()):
                print(f"NaN found in layer {i:02}")            
                breakpoint()
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data)
                # 经过修改没一层神经元的初始值， 避免梯度爆炸
                # 参考 week4 lesson 1
                nn.init.normal_(m.weight.data, std = np.sqrt(1/self.neural_num))

layers = 100
neural_num = 256
batch_size = 16

net = MLP(neural_num, layers)
net.initialize()

inputs = torch.randn((batch_size, neural_num))
output = net(inputs)
print(output)
print("--------------------------------------")
'''
数据为nan， 超出了数据可以表示的范围。
每向前传播一层，标准差扩大 sqrt(n) 倍
'''

# ---- 添加激活函数的 ----
class MLP(nn.Module):
    def __init__(self, neural_num, layers):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.neural_num = neural_num

    def forward(self, x):
        for (i, linear) in enumerate(self.linears):
            x = linear(x)
            # 增加激活函数
            x = torch.tanh(x) # 导致梯度消失
            # 打印每一层的方差
            print(f"layer {i:02}: {x.std()}")
            if torch.isnan(x.std()):
                print(f"NaN found in layer {i:02}")            
                breakpoint()
        return x
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight.data)
                # 参考 week4 lesson 1， 手工计算xavier初始化参数
                a = np.sqrt(6 / (self.neural_num+self.neural_num))
                tanh_gain = nn.init.calculate_gain('tanh')
                '''
                a *= tanh_gain
                nn.init.uniform_(m.weight.data, -a, a)
                '''
                # 利用pytorch自带的xavier初始化
                nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)

layers = 100
neural_num = 256
batch_size = 16

net = MLP(neural_num, layers)
net.initialize()

inputs = torch.randn((batch_size, neural_num))
output = net(inputs)
print(output)

'''
对于饱和激活函数，可以使用 xavier 方法进行初始化
对于非饱和激活函数，需要用到

Relu ---- Kaiming

'''

# ---- calculate gain ----

x = torch.randn(10000)
out = torch.tanh(x)

gain = x.std() / out.std()
print(f"gain: {gain}")

tanh_gain = nn.init.calculate_gain('tanh')
print("tanh_gain in PyTorch:", tanh_gain)

