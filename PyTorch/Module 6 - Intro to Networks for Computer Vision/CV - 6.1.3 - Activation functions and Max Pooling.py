import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc
import torch.nn.functional as F

# 1) Activation Functions

# Create a kernel and image as usual. Set the bias to zero:

conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
Gx=torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0,-1.0]])
conv.state_dict()['weight'][0][0]=Gx
conv.state_dict()['bias'][0]=0.0
conv.state_dict()
image=torch.zeros(1,1,5,5)
image[0,0,:,2]=1
print(image)

# Apply convolution to the image:
Z=conv(image)
print(Z)

# Apply the activation function to the activation map. This will apply the activation function to each element in the activation map.
A=F.relu(Z)
print(A)

# 2) Max Pooling

image1=torch.zeros(1,1,4,4)
image1[0,0,0,:]=torch.tensor([1.0,2.0,3.0,-4.0])
image1[0,0,1,:]=torch.tensor([0.0,2.0,-3.0,0.0])
image1[0,0,2,:]=torch.tensor([0.0,2.0,3.0,1.0])

print(image1)

# Create a maxpooling object in 2d as follows and perform max pooling as follows:

max3=torch.nn.MaxPool2d(2,stride=1)
print(max3(image1))
max1=torch.nn.MaxPool2d(2)
print(max1(image1))

