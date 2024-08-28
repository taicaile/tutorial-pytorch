import torch
import torch.nn as nn
from torchvision.utils import make_grid
torch.manual_seed(1)

# ---- tensor hook ----
# 通过hook函数获得中间运算的梯度

flag = 0
# flag = 1
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w,x)
    b = torch.add(w,1)
    y = torch.mul(a,b)

    a_grad = list()
    
    def grad_hook(grad):
        a_grad.append(grad)

    handle = a.register_hook(grad_hook)

    y.backward()

    # 查看梯度
    print("gradient:", w.grad, x.grad, a.grad, b.grad)
    print("a_grad[0]:", a_grad[0])
    handle.remove()

# ---- tensor hook 直接修改梯度数值 ----

flag = 0
# flag = 1
if flag:
    w = torch.tensor([1.], requires_grad=True)
    x = torch.tensor([2.], requires_grad=True)
    a = torch.add(w,x)
    b = torch.add(w,1)
    y = torch.mul(a,b)

    a_grad = list()
    
    def grad_hook(grad):
        grad*=2
        # return grad*=3, # return 会覆盖掉原有的张量

    handle = w.register_hook(grad_hook)

    y.backward()

    # 查看梯度
    print("gradient:", w.grad)
    handle.remove()

# ---- hook 函数查看网路中间层 ----
flag = 0
flag = 1
if flag:
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1,2,3)
            self.pool1 = nn.MaxPool2d(2,2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.pool1(x)
            return x

    def forward_hook(module, data_input, data_output):
        fmap_block.append(data_output)
        input_block.append(data_input)

    # 前向传播之前运行
    def forward_pre_hook(module, data_input):
        print("forward_pre_hook data_input:{}".format(data_input))
    
    def backward_hook(module, grad_input, grad_output):
        print("backward hook input:{}".format(grad_input))
        print("backward hook output:{}".format(grad_output))

    # 初始化网络
    net = Net()
    net.conv1.weight[0].detach().fill_(1)
    net.conv1.weight[1].detach().fill_(2)
    net.conv1.bias.data.detach().zero_()

    # 注册hook
    fmap_block = list()
    input_block = list()
    net.conv1.register_forward_hook(forward_hook)
    net.conv1.register_forward_pre_hook(forward_pre_hook)
    net.conv1.register_backward_hook(backward_hook)
    # inference
    fake_img = torch.ones((1,1,4,4)) # batchsize * channel * H * W
    output = net(fake_img)

    # loss_fnc = nn.L1Loss()
    # target = torch.randn_like(output)
    # loss = loss_fnc(target, output)
    # loss.backward()

    # 
    print("output shape: {}\n output value:{}\n".format(output.shape, output))
    print("feature maps shape: {}\n output value:{}\n".format(fmap_block[0].shape, fmap_block[0]))
    print("input shape: {}\n input value:{}\n".format(input_block[0][0].shape, input_block[0]))

# ---- feature map virtualization ----
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import torchvision.models as models
from PIL import Image

flag = 0
flag = 1
if flag:
    writer = SummaryWriter()
    # 数据
    path_img = "data/lena.png"
    normMean = [0.5] * 3
    normStd = (0.25) * 3

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

    # 注册hook
    fmap_dict = dict()

    for name, sub_module in alexnet.named_modules():
        # .named_modules() # 迭代返回 name 和 modules
        if isinstance(sub_module, nn.Conv2d):
            key_name = str(sub_module.weight.shape)
            fmap_dict.setdefault(key_name, list())
            print("____", name, sub_module)
            n1, n2 = name.split('.')

            def hook_func(m, i, o):
                key_name = str(m.weight.shape)
                fmap_dict[key_name].append(o)

            alexnet._modules[n1]._modules[n2].register_forward_hook(hook_func)

    output = alexnet(img_tensor)

    # add image
    for layer_name, fmap_list in fmap_dict.items():
        fmap = fmap_list[0]
        fmap.transpose_(0,1)
        
        nrow = int((fmap.shape[0])**0.5)
        fmap_grid = make_grid(fmap, normalize=False, scale_each=True, nrow=nrow)
        writer.add_image("feature map in {}".format(layer_name), fmap_grid, global_step=322)
