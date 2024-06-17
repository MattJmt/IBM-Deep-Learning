# Kernels are used to extract features (ex: edges)
# 3 input channels would be for example the blue, red and green matrices of a picture
# number of kernels depends on input and output channel. If input channel = 2 and output = 3, there are 6 kernels total, [3*2]
# ex: output 1 = k1*x1 + k2* x2 (and so on)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

# Create a Conv2d with three channels:

conv1 = nn.Conv2d(in_channels=1, out_channels=3,kernel_size=3)

# Pytorch randomly assigns values to each kernel. However, use kernels that have been developed to detect edges:

Gx=torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0.0,-1.0]])
Gy=torch.tensor([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])

conv1.state_dict()['weight'][0][0]=Gx
conv1.state_dict()['weight'][1][0]=Gy
conv1.state_dict()['weight'][2][0]=torch.ones(3,3)

# Each kernel has its own bias, so set them all to zero

conv1.state_dict()['bias'][:]=torch.tensor([0.0,0.0,0.0])
print(conv1.state_dict()['bias'])

for x in conv1.state_dict()['weight']:
    print(x)

# Create an input image to represent the input X:
image=torch.zeros(1,1,5,5)
image[0,0,:,2]=1
print(image)

plt.imshow(image[0,0,:,:].numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.colorbar()
plt.show()

# Perform convolution using each channel:

out=conv1(image)

print(out.shape)

for channel,image in enumerate(out[0]):
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

# If you use a different image, the result will be different:

image1=torch.zeros(1,1,5,5)
image1[0,0,2,:]=1
print(image1)
plt.imshow(image1[0,0,:,:].detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
plt.show()

# In this case, the second channel fluctuates, and the first and the third channels produce a constant value.

out1=conv1(image1)
for channel,image in enumerate(out1[0]):
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

# Create an input with two channels:

image2=torch.zeros(1,2,5,5)
image2[0,0,2,:]=-2
image2[0,1,2,:]=1
print(image2)

for channel,image in enumerate(image2[0]):
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

conv3 = nn.Conv2d(in_channels=2, out_channels=1,kernel_size=3)
# Assign kernel values to make the math a little easier:
Gx1=torch.tensor([[0.0,0.0,0.0],[0,1.0,0],[0.0,0.0,0.0]])
conv3.state_dict()['weight'][0][0]=1*Gx1
conv3.state_dict()['weight'][0][1]=-2*Gx1
conv3.state_dict()['bias'][:]=torch.tensor([0.0])

print(conv3.state_dict()['weight'])

# Perform the convolution:

conv3(image2)

# Create an example with two inputs and three outputs and assign the kernel values to make the math a little easier

conv4 = nn.Conv2d(in_channels=2, out_channels=3,kernel_size=3)
conv4.state_dict()['weight'][0][0]=torch.tensor([[0.0,0.0,0.0],[0,0.5,0],[0.0,0.0,0.0]])
conv4.state_dict()['weight'][0][1]=torch.tensor([[0.0,0.0,0.0],[0,0.5,0],[0.0,0.0,0.0]])


conv4.state_dict()['weight'][1][0]=torch.tensor([[0.0,0.0,0.0],[0,1,0],[0.0,0.0,0.0]])
conv4.state_dict()['weight'][1][1]=torch.tensor([[0.0,0.0,0.0],[0,-1,0],[0.0,0.0,0.0]])

conv4.state_dict()['weight'][2][0]=torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0.0,-1.0]])
conv4.state_dict()['weight'][2][1]=torch.tensor([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]])
# For each output, there is a bias, so set them all to zero:
conv4.state_dict()['bias'][:]=torch.tensor([0.0,0.0,0.0])

# Create a two-channel image and plot the results:
image4=torch.zeros(1,2,5,5)
image4[0][0]=torch.ones(5,5)
image4[0][1][2][2]=1
for channel,image in enumerate(image4[0]):
    plt.imshow(image.detach().numpy(), interpolation='nearest', cmap=plt.cm.gray)
    print(image)
    plt.title("channel {}".format(channel))
    plt.colorbar()
    plt.show()

# Perform the convolution:

z=conv4(image4)
print(z)

