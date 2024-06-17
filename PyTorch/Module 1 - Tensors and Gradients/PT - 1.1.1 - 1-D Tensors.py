# to change from numpy to pytorch and back
import torch
import numpy as np
numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
torch_tensor = torch.from_numpy(numpy_array)
back_to_numpy = torch_tensor.numpy()

# to change pandas to pytorch
import pandas as pd
pandas_series = pd.Series([0.1, 2, 3, 10.1])
pandas_to_torch = torch.from_numpy(pandas_series.values)

# return a list to tensor
this_tensor = torch.tensor([1, 2, 3])
torch_to_list = this_tensor.tolist()
torch_to_list: [1, 2, 3]

# Indexing and slicing
c = torch.tensor([0, 1, 2])
print(c)
c[0] = 100
print(c)
c[2] = 5
print(c)
# CY
# tensor([0, 1, 2])
# tensor([100,   1,   2])
# tensor([100,   1,   5])

# Slicing
c = torch.tensor([0, 1, 2, 4])
d = c[1:3]
print(d) # tensor([1, 2])

c = torch.tensor([0, 1, 2, 4])
c[1:3] = torch.tensor([300, 400])
print(c) # tensor([  0, 300, 400,   4])

# BASIC OPERATIONS
# Vector Addition, Substraction, Multiplication, Dot product
u = torch.tensor([1, 2])
v = torch.tensor([2, 3])
z = u + v
print(z) # tensor([3, 5])
z = u * v
print(z) # tensor([2, 6])
z = torch.dot(u,v)
print (z) # tensor(8)

# FUNCTIONS
# mean
a = torch.tensor([1.0, -1.1, 2.0])
mean_a = a.mean()
print(mean_a) # tensor(0.6333)
# max value
max_a = a.max()
print(max_a) # tensor(2.)
# trig
x = torch.tensor([0, np.pi/2, np.pi])
y = torch.sin(x)
print (y.int()) # tensor([0, 1, 0], dtype=torch.int32)
# linspace is to evenly space samples
x = torch.linspace(0, 2*np.pi, 100)
print(x)
# CY
# tensor([0.0000, 0.0635, 0.1269, 0.1904, 0.2539, 0.3173, 0.3808, 0.4443, 0.5077,
#         0.5712, 0.6347, 0.6981, 0.7616, 0.8251, 0.8885, 0.9520, 1.0155, 1.0789,
#         1.1424, 1.2059, 1.2693, 1.3328, 1.3963, 1.4597, 1.5232, 1.5867, 1.6501,
#         1.7136, 1.7771, 1.8405, 1.9040, 1.9675, 2.0309, 2.0944, 2.1579, 2.2213,
#         2.2848, 2.3483, 2.4117, 2.4752, 2.5387, 2.6021, 2.6656, 2.7291, 2.7925,
#         2.8560, 2.9195, 2.9829, 3.0464, 3.1099, 3.1733, 3.2368, 3.3003, 3.3637,
#         3.4272, 3.4907, 3.5541, 3.6176, 3.6811, 3.7445, 3.8080, 3.8715, 3.9349,
#         3.9984, 4.0619, 4.1253, 4.1888, 4.2523, 4.3157, 4.3792, 4.4427, 4.5061,
#         4.5696, 4.6331, 4.6965, 4.7600, 4.8235, 4.8869, 4.9504, 5.0139, 5.0773,
#         5.1408, 5.2043, 5.2677, 5.3312, 5.3947, 5.4581, 5.5216, 5.5851, 5.6485,
#         5.7120, 5.7755, 5.8389, 5.9024, 5.9659, 6.0293, 6.0928, 6.1563, 6.2197,
#         6.2832])
import matplotlib.pyplot as plt
y = torch.cos(x)
plt.plot(x, y)
plt.show() # THIS IS FCKING AWESOME
# naming the x axis
plt.xlabel('x - axis')
# naming the y axis
plt.ylabel('y - axis')

# giving a title to my graph
plt.title('My first graph!')

a=torch.linspace(1, 3, steps=5)
print(a, a.mean()) # tensor([1.0000, 1.5000, 2.0000, 2.5000, 3.0000]) tensor(2.)

# These are the libraries will be used for this lab.

import torch
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
% matplotlib
inline


# Plot vecotrs, please keep the parameters in the same length
# @param: Vectors = [{"vector": vector variable, "name": name of vector, "color": color of the vector on diagram}]

def plotVec(vectors):
    ax = plt.axes()

    # For loop to draw the vectors
    for vec in vectors:
        ax.arrow(0, 0, *vec["vector"], head_width=0.05, color=vec["color"], head_length=0.1)
        plt.text(*(vec["vector"] + 0.1), vec["name"])

    plt.ylim(-2, 2)
    plt.xlim(-2, 2)

    # Convert a integer list with length 5 to a tensor


ints_to_tensor = torch.tensor([0, 1, 2, 3, 4])
print("The dtype of tensor object after converting it to tensor: ", ints_to_tensor.dtype)
print("The type of tensor object after converting it to tensor: ", ints_to_tensor.type())

# Convert a float list with length 5 to a tensor

floats_to_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
print("The dtype of tensor object after converting it to tensor: ", floats_to_tensor.dtype)
print("The type of tensor object after converting it to tensor: ", floats_to_tensor.type())

# Convert a integer list with length 5 to float tensor

new_float_tensor = torch.FloatTensor([0, 1, 2, 3, 4])
new_float_tensor.type()
print("The type of the new_float_tensor:", new_float_tensor.type())

# Another method to convert the integer list to float tensor

old_int_tensor = torch.tensor([0, 1, 2, 3, 4])
new_float_tensor = old_int_tensor.type(torch.FloatTensor)
print("The type of the new_float_tensor:", new_float_tensor.type())

# Introduce the tensor_obj.size() & tensor_ndimension.size() methods

print("The size of the new_float_tensor: ", new_float_tensor.size())

print("The dimension of the new_float_tensor: ", new_float_tensor.ndimension()
# Introduce the tensor_obj.view(row, column) method

twoD_float_tensor = new_float_tensor.view(5, 1)
print("Original Size: ", new_float_tensor)
print("Size after view method", twoD_float_tensor)
# Introduce the use of -1 in tensor_obj.view(row, column) method

twoD_float_tensor = new_float_tensor.view(-1, 1)
print("Original Size: ", new_float_tensor)
print("Size after view method", twoD_float_tensor)

# Convert a numpy array to a tensor

numpy_array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
new_tensor = torch.from_numpy(numpy_array)

print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())

# Convert a tensor to a numpy array

back_to_numpy = new_tensor.numpy()
print("The numpy array from tensor: ", back_to_numpy)
print("The dtype of numpy array: ", back_to_numpy.dtype)

# Set all elements in numpy array to zero
numpy_array[:] = 0
print("The new tensor points to numpy_array : ", new_tensor)
print("and back to numpy array points to the tensor: ", back_to_numpy)

# Convert a panda series to a tensor

pandas_series = pd.Series([0.1, 2, 0.3, 10.1])
new_tensor = torch.from_numpy(pandas_series.values)
print("The new tensor from numpy array: ", new_tensor)
print("The dtype of new tensor: ", new_tensor.dtype)
print("The type of new tensor: ", new_tensor.type())

# Calculate the standard deviation for math_tensor

standard_deviation = math_tensor.std()
