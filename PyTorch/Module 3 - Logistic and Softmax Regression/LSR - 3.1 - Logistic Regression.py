# Logistic Regression is basically linear regression but use sigmoid function on the output
# Import the libraries we need for this lab

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
# Set the random seed

torch.manual_seed(2)

# 1) Logistic Function

z = torch.arange(-100, 100, 0.1).view(-1, 1) # create tensor ranging from -100 to 100
print("The tensor: ", z)

# Create sigmoid object

sig = nn.Sigmoid()

# Use sigmoid object to calculate the

yhat = sig(z)

plt.plot(z.numpy(), yhat.numpy())
plt.xlabel('z')
plt.ylabel('yhat')
yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()

# 2) Build a Logistic Regression with nn.Sequential

# Create x and X tensor

x = torch.tensor([[1.0]])
X = torch.tensor([[1.0], [100]])
print('x = ', x)
print('X = ', X)

# Use sequential function to create model

model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())

# Print the parameters

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# The prediction for x

yhat = model(x)
print("The prediction: ", yhat)

# The prediction for X

yhat = model(X)
print(yhat)

# Create and print samples

x = torch.tensor([[1.0, 1.0]]) # 1*2 tensor
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]) # 3*2 tensor
print('x = ', x)
print('X = ', X)

# Create new model using nn.sequential()

model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())

# Print the parameters

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# Make the prediction of x

yhat = model(x)
print("The prediction: ", yhat)

# The prediction of X

yhat = model(X)
print("The prediction: ", yhat)

# 3) Build Custom Modules (Alternative to nn.sequential) - model or object function is identical to using nn.Sequential

# Create logistic_regression custom class

class logistic_regression(nn.Module):

    # Constructor
    def __init__(self, n_inputs):
        super(logistic_regression, self).__init__()
        self.linear = nn.Linear(n_inputs, 1)

    # Prediction
    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat

# Create x and X tensor

x = torch.tensor([[1.0]])
X = torch.tensor([[-100], [0], [100.0]])
print('x = ', x)
print('X = ', X)

# Create logistic regression model

model = logistic_regression(1)

# Print parameters

print("list(model.parameters()):\n ", list(model.parameters()))
print("\nmodel.state_dict():\n ", model.state_dict())

# Make the prediction of x

yhat = model(x)
print("The prediction result: \n", yhat)

# Make the prediction of X

yhat = model(X)
print("The prediction result: \n", yhat)

# Create logistic regression model

model = logistic_regression(2)

# Create x and X tensor

x = torch.tensor([[1.0, 2.0]])
X = torch.tensor([[100, -100], [0.0, 0.0], [-100, 100]])
print('x = ', x)
print('X = ', X)

# Make the prediction of x

yhat = model(x)
print("The prediction result: \n", yhat)

# Make the prediction of X

yhat = model(X)
print("The prediction result: \n", yhat)