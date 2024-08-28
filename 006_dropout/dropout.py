'''
target: design a method to test dropout performance
'''

import torch
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms

# dropout function test

# build model MLP
class MLP(nn.Module):
    def __init__(self, input=1*28*28, output=10):
        super().__init__()
        self.input = input
        self.fc1 = nn.Linear(input, 20)
        self.fc2 = nn.Linear(20,40)
        self.fc3 = nn.Linear(40, output)
        self.relu = nn.ReLU()
        self.output = output

    def forward(self,x):
        x = F.dropout(self.relu(self.fc1(x.view(-1,self.input))))
        x = F.dropout(self.relu(self.fc2(x)))
        # x = self.relu(self.fc1(x.view(-1,self.input)))
        # x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

traindata = torchvision.datasets.MNIST(train=True, download=True, root="pytorch_data", transform=trans)
validata = torchvision.datasets.MNIST(train=False, download=True, root="pytorch_data", transform=trans)

train_loader = dataloader.DataLoader(traindata, batch_size=64, shuffle=True)
val_loader = dataloader.DataLoader(validata, batch_size=64, shuffle=False)

example_data, example_targets = next(iter(train_loader))
print(example_data.shape)
print(example_targets.shape)
print(traindata.classes)

model = MLP()
model.train()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# criteria = nn.NLLLoss() # if forward return F.log_softmax(x)
criteria = nn.CrossEntropyLoss() # ->log_softmax()->nn.NLLLoss()

for epoch in range(10):
    running_loss = 0.0
    running_corrects = 0
    nsamples = 0

    for idx, (data, target) in enumerate(train_loader):
        outputs = model(data)
        _, preds = torch.max(outputs,dim=1)
        loss = criteria(outputs, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_corrects+=(preds==target).sum()
        running_loss+=loss.item()
        nsamples+=len(target)

    print(epoch, f"{running_loss/nsamples:0.5f}", f"{(running_corrects/nsamples).item():0.2f}")

model.eval()
with torch.no_grad():
    for idx, (data, target) in enumerate(val_loader):
        outputs = model(data)
        _, preds = torch.max(outputs,dim=1)
        running_corrects+=(preds==target).sum()
        nsamples+=len(target)
    print(f"{(running_corrects/nsamples).item():0.2f} (val)")


''' without dropout
0 0.01323 0.77
1 0.00516 0.90
2 0.00431 0.92
3 0.00383 0.93
4 0.00348 0.94
5 0.00323 0.94
6 0.00301 0.94
7 0.00285 0.95
8 0.00270 0.95
9 0.00258 0.95
0.95 (val)
'''

''' with dropout
0 0.02707 0.37
1 0.01833 0.58
2 0.01609 0.64
3 0.01495 0.68
4 0.01418 0.70
5 0.01365 0.71
6 0.01332 0.72
7 0.01295 0.73
8 0.01276 0.73
9 0.01263 0.74
0.74 (val)
'''