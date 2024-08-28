import os
import time
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data.sampler import BatchSampler
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
import torchvision.models as models
torch.manual_seed(1)

rmb_label = {"1":0, "100":1}

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None) -> None:
        super().__init__()
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

flag = 0
# flag = 1
if flag:
    writer = SummaryWriter()
    # image 1
    fake_img = torch.randn(3,512,512)
    writer.add_image("fake_img", fake_img, 1)
    time.sleep(1)

    # image 2
    fake_img = torch.ones(3,512,512)
    writer.add_image("fake_img", fake_img, 2)
    time.sleep(1)

    # image 3
    fake_img = torch.ones(3,512,512) * 1.1
    writer.add_image("fake_img", fake_img, 3)
    time.sleep(1)

    # image 4 HW
    fake_img = torch.rand(512, 512)
    writer.add_image("fake_img", fake_img, 4, dataformats="HW")
    time.sleep(1)

    # image 4 HWC
    fake_img = torch.randn(512, 512, 3)
    writer.add_image("fake_img", fake_img, 5, dataformats="HWC")
    time.sleep(1)

    writer.close()

    # ---- make_grid ----
    flag = 1
    if flag:
        writer = SummaryWriter()

        split_dir = os.path.join("data", "rmb_split")
        train_dir = os.path.join(split_dir, "train")

        transform_compose = transforms.Compose([
            transforms.Resize((32, 64)),
            transforms.ToTensor()
        ])

        train_data = RMBDataset(train_dir, transform_compose)
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        data_batch, data_label = next(iter(train_loader))

        img_grid = make_grid(data_batch, normalize=False, scale_each=False, nrow=4)
        writer.add_image('img_grid',img_grid,0)
        writer.close()

# ---- kernel visualization ----

flag = 0
# flag = 1

if flag:
    writer = SummaryWriter()
    alexnet = models.alexnet(pretrained=True)
    kernel_num = -1
    vis_max = 1

    for sub_module in alexnet.modules():
        if isinstance(sub_module, nn.Conv2d):
            kernel_num+=1
            if kernel_num>vis_max:
                break
            kernels = sub_module.weight
            c_out, c_int, k_w, k_h = tuple(kernels.shape)
            print(tuple(kernels.shape))
            for o_idx in range(c_out):
                kernel_idx = kernels[o_idx,:,:,:].unsqueeze(1)
                print(kernel_idx.shape)
                kernel_grid = make_grid(kernel_idx, normalize=False,scale_each=True, nrow=c_int)
                writer.add_image(f"{kernel_num} convlayer split_in_channel", kernel_grid, global_step=o_idx)
                print(f"{kernel_num} convlayer split_in_channel")

            kernel_all = kernels.view(-1, 3, k_h, k_w)
            kernel_grid = make_grid(kernel_all, normalize=False, scale_each=True, nrow=8)
            writer.add_image(f"{kernel_num}_all", kernel_grid, global_step=322)

            print(f"{kernel_num} convlayer shape:{tuple(kernels.shape)}")

    writer.close()

# ----
flag = 0
# flag = 1

if flag:
    writer = SummaryWriter()

    path_img = "data/lena.png"
    normMean = [0.5, 0.5, 0.5]
    normStd = [0.25, 0.25, 0.25]

    norm_transform = transforms.Normalize(normMean, normStd)
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        norm_transform
    ])

    img_pil = Image.open(path_img).convert('RGB')
    if img_transforms is not None:
        img_tensor = img_transforms(img_pil)
    img_tensor.unsqueeze_(0)

    # 模型
    alexnet = models.alexnet(pretrained=True)

    # forward
    convlayer1 = alexnet.features[0]
    fmap_1 = convlayer1(img_tensor)

    # 预处理
    fmap_1.transpose_(0,1)
    fmap_1_grad = make_grid(fmap_1, normalize=False, scale_each=True, nrow=8)

    writer.add_image("feature map in conv1", fmap_1_grad, global_step=322)
    writer.close()

# ---- add_graph ----
import torch.nn.functional as F
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

flag = 0
# flag = 1
if flag:
    writer = SummaryWriter()
    
    fake_img = torch.randn(1,3,32,32)
    lenet = LeNet(classes=2)
    writer.add_graph(lenet, fake_img)
    writer.close()

# ---- torchsummary ----
# pip install torchsummary
flag = 0
flag = 1
if flag:
    from torchsummary import summary
    lenet = LeNet(classes=2)
    print(summary(lenet, input_size=(3,32,32), device='cpu'))