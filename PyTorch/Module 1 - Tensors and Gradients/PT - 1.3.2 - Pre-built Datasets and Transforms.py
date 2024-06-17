# These are the libraries will be used for this lab.

import torch
import matplotlib.pylab as plt
import numpy as np
torch.manual_seed(0)

# Show data by diagram

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title("y = {}".format(data_sample[1]))

# PREBUILT DATASETS
# Run the command below when you do not have torchvision installed
# !conda install -y torchvision

import torchvision.transforms as transforms
import torchvision.datasets as dsets
# Import the prebuilt dataset into variable dataset

dataset = dsets.MNIST(
    root = './data',
    train = False,
    download = True,
    transform = transforms.ToTensor()
)
# Examine whether the elements in dataset MNIST are tuples, and what is in the tuple?

print("Type of the first element: ", type(dataset[0]))
print("The length of the tuple: ", len(dataset[0]))
print("The shape of the first element in the tuple: ", dataset[0][0].shape)
print("The type of the first element in the tuple", type(dataset[0][0]))
print("The second element in the tuple: ", dataset[0][1])
print("The type of the second element in the tuple: ", type(dataset[0][1]))
print("As the result, the structure of the first element in the dataset is (tensor([1, 28, 28]), tensor(7)).")

# Plot the first element in the dataset

show_data(dataset[0]) # works in jupyter labs

# Plot the second element in the dataset

show_data(dataset[1])

# TORCHVISION TRANSFORMS (flips image of 2 (output image) and crops the tensor
# Combine two transforms: crop and convert to tensor. Apply the compose to MNIST dataset

croptensor_data_transform = transforms.Compose([transforms.CenterCrop(20), transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = croptensor_data_transform)
print("The shape of the first element in the first tuple: ", dataset[0][0].shape)
# Plot the first element in the dataset

show_data(dataset[0],shape = (20, 20))
# Plot the second element in the dataset

show_data(dataset[1],shape = (20, 20))
# Construct the compose. Apply it on MNIST dataset. Plot the image out.

fliptensor_data_transform = transforms.Compose([transforms.RandomHorizontalFlip(p = 1),transforms.ToTensor()])
dataset = dsets.MNIST(root = './data', train = False, download = True, transform = fliptensor_data_transform)
show_data(dataset[1])