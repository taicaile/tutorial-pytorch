import random
from PIL import Image
import numpy as np
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(),
    transforms.Resize(),
    transforms.RandomSizedCrop(),
    transforms.RandomErasing(),
    transforms.Lambda(),
    transforms.Grayscale(),
    transforms.RandomGrayscale(),
    transforms.RandomAffine()
])

# Transform Operation
transforms.RandomChoice() # 随机选择一个transforms方法
transforms.RandomApply() # 根据概率执行一组transforms方法
transforms.RandomOrder() # 对一组transforms随机排序


# 自定义transforms方法
# 椒盐噪声

class AddPepperNoise():
    def __init__(self, snr, p=0.9) -> None:
        assert isinstance(snr, float) and isinstance(p, float)
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL Image
        """
        if random.uniform(0,1) < self.p:
            img_ = np.array(img).copy()
            h,w,c = np.shape
            signal_pct = self.snr
            noise_pct = (1-self.snr)
            mask = np.random.choice((0,1,2), size=(h,w,1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask==1] = 255 # 盐噪声
            img_[mask==2] == 0  # 椒噪声
            return Image.fromarray(img_.astype('uint8')).convert('RGB')
        else:
            return img



transform = transforms.Compose([transforms.ToTensor()])
# call transforms.Compose __call__ method
# transform(img)
