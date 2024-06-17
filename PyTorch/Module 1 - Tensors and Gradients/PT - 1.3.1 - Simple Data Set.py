# These are the libraries will be used for this lab.
import torch
from torch.utils.data import Dataset
torch.manual_seed(1)

# Define class for dataset
class toy_set(Dataset):

    # Constructor with defult values
    def __init__(self, length=100, transform=None):
        self.len = length
        self.x = 2 * torch.ones(length, 2)
        self.y = torch.ones(length, 1)
        self.transform = transform

    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len

# Create Dataset Object. Find out the value on index 1. Find out the length of Dataset Object.
our_dataset = toy_set()
print("Our toy_set object: ", our_dataset)
print("Value on index 0 of our toy_set object: ", our_dataset[0])
print("Our toy_set length: ", len(our_dataset))

# Use loop to print out first 3 elements in dataset
for i in range(3):
    x, y=our_dataset[i]
    print("index: ", i, '; x:', x, '; y:', y)
# CY:
# Our toy_set object:  <__main__.toy_set object at 0x7f9a6163c518>
# Value on index 0 of our toy_set object:  (tensor([2., 2.]), tensor([1.]))
# Our toy_set length:  100
# index:  0 ; x: tensor([2., 2.]) ; y: tensor([1.])
# index:  1 ; x: tensor([2., 2.]) ; y: tensor([1.])
# index:  2 ; x: tensor([2., 2.]) ; y: tensor([1.])

# TRANSFORMS
# Create tranform class add_mult

class add_mult(object):

    # Constructor
    def __init__(self, addx=1, muly=2):
        self.addx = addx
        self.muly = muly

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x + self.addx
        y = y * self.muly
        sample = x, y
        return sample
# Create an add_mult transform object, and an toy_set object

a_m = add_mult()
data_set = toy_set()

# Use loop to print out first 10 elements in dataset
for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = a_m(data_set[i])
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)

# Create a new data_set object with add_mult object as transform
cust_data_set = toy_set(transform = a_m)

# Use loop to print out first 10 elements in dataset
for i in range(10):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
# Index:  0 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  0 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  1 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  1 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  2 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  2 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  3 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  3 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  4 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  4 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  5 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  5 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  6 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  6 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  7 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  7 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  8 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  8 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  9 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  9 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  0 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  0 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  1 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  1 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  2 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  2 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  3 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  3 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  4 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  4 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  5 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  5 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  6 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  6 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  7 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  7 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  8 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  8 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  9 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  9 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])

# COMPOSE
# Run the command below when you do not have torchvision installed
# !conda install -y torchvision
from torchvision import transforms

# Create tranform class mult

class mult(object):

    # Constructor
    def __init__(self, mult=100):
        self.mult = mult

    # Executor
    def __call__(self, sample):
        x = sample[0]
        y = sample[1]
        x = x * self.mult
        y = y * self.mult
        sample = x, y
        return sample

# Combine the add_mult() and mult()
data_transform = transforms.Compose([add_mult(), mult()])
print("The combination of transforms (Compose): ", data_transform)

# Create a new toy_set object with compose object as transform
compose_data_set = toy_set(transform = data_transform)

# Use loop to print out first 3 elements in dataset
for i in range(3):
    x, y = data_set[i]
    print('Index: ', i, 'Original x: ', x, 'Original y: ', y)
    x_, y_ = cust_data_set[i]
    print('Index: ', i, 'Transformed x_:', x_, 'Transformed y_:', y_)
    x_co, y_co = compose_data_set[i]
    print('Index: ', i, 'Compose Transformed x_co: ', x_co ,'Compose Transformed y_co: ',y_co)

# The combination of transforms (Compose):  Compose(
#     <__main__.add_mult object at 0x7f99d5f1dda0>
#     <__main__.mult object at 0x7f99d5f4a2b0>
# )
# Index:  0 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  0 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  0 Compose Transformed x_co:  tensor([300., 300.]) Compose Transformed y_co:  tensor([200.])
# Index:  1 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  1 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  1 Compose Transformed x_co:  tensor([300., 300.]) Compose Transformed y_co:  tensor([200.])
# Index:  2 Original x:  tensor([2., 2.]) Original y:  tensor([1.])
# Index:  2 Transformed x_: tensor([3., 3.]) Transformed y_: tensor([2.])
# Index:  2 Compose Transformed x_co:  tensor([300., 300.]) Compose Transformed y_co:  tensor([200.])