import torch
import torchvision
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

alexnet = torchvision.models.alexnet(pretrained=True)

#Updating the second classifier
alexnet.classifier[4] = nn.Linear(4096,1024)
#Updating the third and the last classifier that is the output layer of the network. Make sure to have 10 output nodes if we are going to get 10 class labels through our model.
alexnet.classifier[6] = nn.Linear(1024,10)

alexnet.to(device)
transforms = torchvision.transforms.Compose([
  torchvision.transforms.Resize(256),
  torchvision.transforms.CenterCrop(224),
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

traindata = torchvision.datasets.CIFAR10(root='../data',train=True,transform=transforms, download=True)
validdata = torchvision.datasets.CIFAR10(root='../data',train=False, transform=transforms, download=True)

trainloader = DataLoader(traindata, shuffle=True, batch_size=64)
validloader = DataLoader(validdata, shuffle=False, batch_size=64)

image, label = next(iter(trainloader))
print(image.shape, label)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(alexnet.parameters(),lr=0.001, momentum=0.8)

for epoch in range(30):
  running_loss = 0.0
  correct = 0
  total = 0
  for i, (image,label) in enumerate(trainloader):
    predict = alexnet(image.to(device)) # alexnet expect input size [64, 3, 11, 11]
    loss = criterion(predict, label.to(device))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    running_loss+=loss.item()
    _, p = torch.max(predict, dim=1)
    correct += sum(p.cpu().numpy()==label.numpy())
    total+=len(label)
    if i%200==199:
      print(epoch, i, f"{running_loss/200:0.3f}", f"acc:{correct/total:0.3f}")
      running_loss = 0.0