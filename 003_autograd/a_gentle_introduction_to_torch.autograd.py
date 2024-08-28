
# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

import torch, torchvision
import torch.optim as optim
####################################################
# get the pretraind model
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

# predict label by random input data, forward pass
prediction = model(data)
loss = (prediction - labels).sum()

# backward propagation, Autograd then calcuates and stores the gradients for each 
# model parameters in the parameter's '''.grad''' attribute
loss.backward()

# register all the model's parameters
optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
# we call .step() to initiate gradient descent
optim.step()

####################################################
# Differentiation in Autograd
import torch

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

# create another tensor from a and b, Q=3*a^3-b^2
Q = 3*a**3 - b**2
# \frac{\partial Q}{\partial a} = 9a^2
# \frac{\partial Q}{\partial b} = -2b
# Q = tensor([-24.,  11.], grad_fn=<SubBackward0>)
# Q is a vector
Q.sum().backward()  # or Q.backward(gradient=torch.tensor([1., 1.]))
print(a.grad == 9*a**2) # Q对a的梯度
print(b.grad == -2*b) # Q对b的梯度

# Q is not a leaf Tensor

#############################################
# DAG, in this DAG, leaves are the input tensors, roots are the output tensors.

x = torch.rand(5, 5)
y = torch.rand(5, 5)
z = torch.rand((5, 5), requires_grad=True)

a = x + y
print(f"Does `a` require gradients? : {a.requires_grad}")
b = x + z
print(f"Does `b` require gradients?: {b.requires_grad}")

#############################################
# frozen parameters
from torch import nn, optim
 
model = torchvision.models.resnet18(pretrained=True)

for para in model.parameters():
    para.requires_grad = False

# change a layer
# print(model.fc) # Linear(in_features=512, out_features=1000, bias=True)
model.fc = nn.Linear(512, 10)

# optimizer only the classifier
optimizer = optim.SGD(model.fc.parameters(), lr=1e-2, momentum=0.9)

