from torch import nn
import torch
torch.manual_seed(1)

class linear_regression(nn.Module):
    def __init__(self,input_size,output_size):
        super(linear_regression,self).__init__()
        self.linear=nn.Linear(input_size,output_size) # Applies a linear transformation to the incoming data: y = xA^T + b. Its parameters are - in_features: size of each input sample - out_features:  size of each output sample - bias: If set to False, the layer will not learn an additive bias. Default: True
    def forward(self,x):
        yhat=self.linear(x)
        return yhat

model=linear_regression(1,10)
model(torch.tensor([1.0]))

list(model.parameters()) # see parameters

x=torch.tensor([[1.0],[3.0]])
yhat=model(x)
print(yhat)

X=torch.tensor([[1.0],[2.0],[3.0]])
Yhat=model(X)
print(Yhat)
