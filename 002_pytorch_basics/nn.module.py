from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.modules.linear import Linear

import torchvision

# Containers

# nn.Sequetial

# nn.ModuleList

# nn.ModuleDict

fake_img = torch.randn((4,10,32,32))

#----Sequantial

class LeNetSequential(nn.Module):
    def __init__(self, classes):
        super(LeNetSequential, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,6,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6,16,5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(85, classes),)


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

class LeNetSequentialOrderDict(nn.Module):
    def __init__(self, classes):
        super(LeNetSequentialOrderDict, self).__init__()

        self.features = nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3,6,5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),

            'conv2': nn.Conv2d(6,16,5),
            'relu2': nn.ReLU(inplace=True),
            'pool2': nn.MaxPool2d(kernel_size=2, stride=2)
        }))

        self.classifier = nn.Sequential(OrderedDict({
            'fc1': nn.Linear(16*5*5, 120),
            'relu3': nn.ReLU(),

            'fc2': nn.Linear(120,84),
            'relu4': nn.Relu(),

            'fc3': nn.Linear(84, classes),
         }))

class ModuleList(nn.Module):
    def __init__(self):
        super(ModuleList, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10) for _ in range(20)])

    def forward(self,x):
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x

class ModuleDict(nn.Module):
    def __init__(self):
        super().__init__()
        self.choices = nn.ModuleDict({
            'conv': nn.Conv2d(10,10,3),
            'pool': nn.MaxPool2d(3)
        })

        self.activations = nn.ModuleDict({
            'relu': nn.ReLU(),
            'prelu': nn.PReLU()
        })

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x

net = ModuleDict()
output = net(fake_img, 'conv', 'prelu')

# check AlexNet definition
alexnet = torchvision.models.AlexNet()