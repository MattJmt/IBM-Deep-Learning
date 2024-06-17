import torch
w = torch.tensor(2.0, requires_grad=True) # weight
b = torch.tensor(-1.0, requires_grad=True) # bias

def forward(x): # define the forward function
    y = w*x + b
    return y
x = torch.tensor([[1.0], [2.0]])
yhat = forward(x)
print(yhat) # tensor([[1.],
            # [3.]], grad_fn=<AddBackward0>)

# CLASS LINEAR

from torch.nn import Linear
torch.manual_seed(1) # fixed value of w and b
model = Linear(in_features=1, out_features=1) # create a model so we can make a prediction
print(list(model.parameters()))
# [Parameter containing:
# tensor([[0.5153]], requires_grad=True), Parameter containing:
# tensor([-0.4414], requires_grad=True)]

# we make a prediction
x = torch.tensor([[1.0], [2.0]])
yhat = model(x)
print(yhat)
# tensor([[0.0739],
#         [0.5891]], grad_fn=<AddmmBackward>)

# CUSTOM MODULES
import torch.nn as nn
class LR(nn.Module):
    def __init__(self, in_size, out_size):
        super(LR,self).__init__()
        self.linear=nn.Linear(in_size, out_size)

    def forward(self, x):
        out = self.linear(x)
        return out

model= LR(1,1)  # create a model
print(list(model.parameters()))
# [Parameter containing:
# tensor([[-0.1939]], requires_grad=True), Parameter containing:
# tensor([0.4694], requires_grad=True)2In
