# Import the libraries we need for this lab

import torch.nn as nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
torch.manual_seed(2)

# 1) Logistic Function

# Create a tensor

z = torch.arange(-10, 10, 0.1).view(-1, 1)

# Create a sigmoid object

sig = nn.Sigmoid()

# Make a prediction of sigmoid function

yhat = sig(z)

# Plot the result

plt.plot(z.numpy(),yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')

# Use the build in function to predict the result

yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

# 2) Tanh

# Create a tanh object

TANH = nn.Tanh()

# Make the prediction using tanh object

yhat = TANH(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

# Make the prediction using the build-in tanh object

yhat = torch.tanh(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

#) 3) ReLU

# Create a relu object and make the prediction

RELU = nn.ReLU()
yhat = RELU(z)
plt.plot(z.numpy(), yhat.numpy())

# Use the build-in function to make the prediction

yhat = F.relu(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

#4) Compare activation functions

# Plot the results to compare the activation functions

x = torch.arange(-2, 2, 0.1).view(-1, 1)
plt.plot(x.numpy(), F.relu(x).numpy(), label='relu')
plt.plot(x.numpy(), torch.sigmoid(x).numpy(), label='sigmoid')
plt.plot(x.numpy(), torch.tanh(x).numpy(), label='tanh')
plt.legend()
plt.show()

