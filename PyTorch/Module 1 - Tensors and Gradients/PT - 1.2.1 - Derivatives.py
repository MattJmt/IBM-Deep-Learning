import torch

# SIMPLE DERIVATIVES
x = torch.tensor(2.0, requires_grad = True)
print(x)
y = x**3 # function
print(y)
print("y.backward", y.backward()) # calculates derivative when x=2. only used to update x.grad. output is None. ie. 3x^2
print("x.grad",x.grad) # tensor(4.)

z = x**2 + 2*x + 1
z.backward()
print(x.grad) # tensor(10.)

# PARTIAL DERIVATIVES
u = torch.tensor(1.0, requires_grad=True)
v = torch.tensor(2.0, requires_grad=True)
f = u*v + u**2
f.backward()
print(u.grad, v.grad) # tensor(4.) tensor(1.)

# COOL HACK
import matplotlib.pyplot as plt
x = torch.linspace(-10.0, 10.0, 2, requires_grad = True)# from -10 to 10 every integer '1' value
Y = x**2
y = torch.sum(x**2)
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function') # detach().numpy to convert to numpy array
plt.plot(x.detach().numpy(), x.grad.detach().numpy(), label = 'derivative')
plt.legend()
plt.show()

# ReLU works
import torch.nn.functional as F
# Take the derivative of Relu with respect to multiple value. Plot out the function and its derivative

x = torch.linspace(-3.0, 3.0, 100, requires_grad = True)
Y = F.relu(x)
y = Y.sum()
y.backward()
plt.plot(x.detach().numpy(), Y.detach().numpy(), label = 'function')
plt.plot(x.detach().numpy(), x.grad.numpy(), label = 'derivative')
plt.xlabel('x')
plt.legend()
plt.show()


