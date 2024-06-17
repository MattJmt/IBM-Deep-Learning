# MLR is basically 2 x's (input, or more) joining in the same node.

# Import the libraries and set the random seed

from torch import nn
import torch
torch.manual_seed(1)

# 1) Prediction

# Set the weight and bias

w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)

# Define Prediction Function

def forward(x):
    yhat = torch.mm(x, w) + b # torch.mm = matrix multiplication
    return yhat

# Calculate yhat

x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
print("The result: ", yhat)

# 2) Each row of the following tensor represents a sample:

# Sample tensor X

X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

# Make the prediction of X

yhat = forward(X)
print("The result: ", yhat)

# 3) Class Linear

# Make a linear regression model using build-in function

model = nn.Linear(2, 1)

# Make a prediction of x

yhat = model(x)
print("The result: ", yhat)

# Make a prediction of X

yhat = model(X)
print("The result: ", yhat)

# 4) Build Custom Modules

# Create linear_regression Class

class linear_regression(nn.Module):

    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    # Prediction function
    def forward(self, x):
        yhat = self.linear(x)
        return yhat

model = linear_regression(2, 1) # Build a linear regression object. The input feature size is two.

# Print model parameters

print("The parameters: ", list(model.parameters()))
# Or use
print("The parameters: ", model.state_dict())

# Make a prediction of x

yhat = model(x)
print("The result: ", yhat)

# Make a prediction of X

yhat = model(X)
print("The result: ", yhat)