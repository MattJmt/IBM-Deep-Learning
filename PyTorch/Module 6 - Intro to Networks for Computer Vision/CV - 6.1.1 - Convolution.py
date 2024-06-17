import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
print(conv)

conv.state_dict()['weight'][0][0]=torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0.0,-1.0]])
conv.state_dict()['bias'][0]=0.0
conv.state_dict()

# Create a dummy tensor to represent an image. The shape of the image is (1,1,5,5) where:
# (number of inputs, number of channels, number of rows, number of columns )

image=torch.zeros(1,1,5,5)
image[0,0,:,2]=1
print(image)

z=conv(image)
print(z)

# 2) Determining the Size of the Output
# Create a kernel of size 2:

K=2
conv1 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=K)
conv1.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv1.state_dict()['bias'][0]=0.0
conv1.state_dict()
print(conv1)

# Create an image of size 2:
M=4
image1=torch.ones(1,1,M,M)

# Perform the convolution and verify the size is correct:
z1=conv1(image1)
print("z1:",z1)
print("shape:",z1.shape[2:4])

# 3) Stride parameter

# Create a convolution object with a stride of 2:

conv3 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=2)

conv3.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv3.state_dict()['bias'][0]=0.0
conv3.state_dict()

z3=conv3(image1)

print("z3:",z3)
print("shape:",z3.shape[2:4])

# Zero Padding
# Try performing convolutions with the kernel_size=2 and a stride=3:

conv4 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=3)
conv4.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv4.state_dict()['bias'][0]=0.0
conv4.state_dict()
z4=conv4(image1)
print("z4:",z4)
print("z4:",z4.shape[2:4])


conv5 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=3,padding=1)
conv5.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv5.state_dict()['bias'][0]=0.0
conv5.state_dict()
z5=conv5(image1)
print("z5:",z5)
print("z5:",z5.shape[2:4])