import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import numpy as np
from PIL import Image


torch.manual_seed(0)
rmb_label = {"1":0, "100":0}

# 参数设置
MAX_EPOCH = 10
BATCH_SIZE = 16
LR = 0.01

log_interval = 10
val_interval =- 1

# ---- step 1 数据 ----

split_dir = os.path.join("data","rmb_split")
train_dir = os.path.join(split_dir, "train")
valid_dir = os.path.join(split_dir, "valid")

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomGrayscale(p=0.8),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
        self.label_name = {"1":0, "100":1}
        self.data_info = self.get_img_info(data_dir)
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_info)


    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                
                for i in range(len(img_names)):
                    image_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, image_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))
        return data_info

# 构建 Dataset
train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
valid_data = RMBDataset(data_dir=valid_dir, transform=valid_transform)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)

# ---- 模型 ----
class LeNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.relu(self.fc3(x))


net = LeNet(classes=2)
# net.initialize_weights()

# ---- 损失函数 ----
criterion = nn.CrossEntropyLoss()

# ---- 优化器 ----
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ---- 训练 ----
train_curve = list()
valid_curve = list()

iter_count = 0

# 构建 SummaryWriter
writer = SummaryWriter()

for epoch in range(MAX_EPOCH):
    loss_mean = 0.
    correct = 0.
    total = 0.
    net.train()
    for i, data in enumerate(train_loader):
        iter_count += 1
        inputs, labels = data
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).squeeze().sum().numpy()

        # 打印训练信息
        loss_mean += loss.item()
        train_curve.append(loss.item())
        if (i+1) % log_interval == 0:
            loss_mean = loss_mean / log_interval
            print(f"Train epoch:[{epoch:0>3}/{MAX_EPOCH:0>3}], Iteration:[{i:0>3}/{len(train_loader):0>3}], Loss:{loss_mean:.4f}, Acc:{correct/total:.4f}")
            loss_mean = 0
        
        # 记录数据
        writer.add_scalars("Loss", {'Train':loss.item()}, iter_count)
        writer.add_scalars("Accuracy", {"Train":correct/total}, iter_count)

    # writer, 记录梯度
    for name, layer in net.named_parameters():
        writer.add_histogram(name+"_grad", layer.grad, epoch)
        writer.add_histogram(name+"_data", layer, epoch)
    
    scheduler.step()
    
    # validate the model
    if (epoch+1) % val_interval == 0:

        correct_val = 0.
        total_val = 0.
        loss_val = 0.
        net.eval()

        with torch.no_grad():
            for j, param in enumerate(valid_loader):
                inputs, labels = param
                outputs = net(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted==labels).squeeze().sum().numpy()

                loss_val += loss.item()
            
            valid_curve.append(loss.item())
            print(f"Valid epoch:[{epoch:0>3}/{MAX_EPOCH:0>3}], Iteration:[{i:0>3}/{len(valid_loader):0>3}], Loss:{loss_val:.4f}, Acc:{correct_val/total_val:.4f}")

            # 记录数据，保存于 event file
            writer.add_scalars("Loss", {"valid":np.mean(valid_curve)}, iter_count)
            writer.add_scalars("Acc", {"valid":correct_val/total_val}, iter_count)

train_x = range(len(train_curve))
train_y = train_curve

train_iters = len(train_loader)
