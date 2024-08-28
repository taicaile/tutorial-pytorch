# 直接引用torch, IntelliSense coundn't find nn module
import torch
print(torch.nn.L1Loss())

# import torch.nn, IntelliSense is able to infer L1Loss function
import torch.nn as nn
nn.L1Loss()

# for regression
# Mean Absolute Error, L1 loss function
# loss(x,y) = |x-y|
nn.L1Loss()

# Mean square error
# loss(x,y) = (x-y)^2
nn.MSELoss()

# Smooth L1 Loss
nn.SmoothL1Loss()

## for classification

#  negative log-likelihood loss
# loss(x,y) = -(log y)
nn.NLLLoss()

# cross entrypy loss
nn.CrossEntropyLoss()

# # Binary Cross Entrypy
nn.BCELoss()