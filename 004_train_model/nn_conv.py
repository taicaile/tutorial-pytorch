import os
import torch
import torch.nn as nn

import numpy as np

from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

torch.manual_seed(1)

# ---- load image ----
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
img_path = os.path.join(BASE_DIR, 'data','lena.png')
img = Image.open(img_path).convert('RGB') # height x width x channel

# ---- convert to tensor ----
img_transform = transforms.Compose([transforms.ToTensor()]) # totensor , c w h
img_tensor = img_transform(img)
img_tensor.unsqueeze_(dim=0) # batchsize c w h

# ---- create convolution layer ----
conv_layer = nn.Conv2d(3,1,3) # input, output, kernel_size
nn.init.xavier_normal_(conv_layer.weight.data)
img_conv = conv_layer(img_tensor)
# ---- transposed ----


# ---- transform inverse ----
def transform_invert(img, transforms):
    """
    将输入的tensor进行反transform操作
    :param img: tensor
    :param transforms: torchvision.transforms.Compose()
    :return: PIL Image
    """
    
    if 'Normalize' in str(transforms):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transforms.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img.dtype, device=img.device)
        std = torch.tensor(norm_transform[0].std, dtype=img.dtype, device=img.device)
        img.mul_(std[:, None, None]).add_(mean[:, None, None])  # 乘方差 加均值

    img = img.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    if 'ToTensor' in str(transforms):
        img = img.detach() # 添加该行密码
        img = np.array(img) * 255

    # 将numpy_array转化为PIL
    if img.shape[2] == 3:
        img = Image.fromarray(img.astype('uint8')).convert('RGB')
    elif img.shape[2] == 1:
        img = Image.fromarray(img.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img.shape[2]) )

    return img

# ---- visualization----
print("卷积前尺寸:{}\n卷积后尺寸:{}".format(img_tensor.shape, img_conv.shape))

img_conv = transform_invert(img_conv[0,0:1,...], img_transform) # 经过一次卷积操作后的图像
img_raw = transform_invert(img_tensor.squeeze(), img_transform) # 原图像

plt.subplot(122).imshow(img_conv, cmap='gray')
plt.subplot(121).imshow(img_raw)
plt.show()