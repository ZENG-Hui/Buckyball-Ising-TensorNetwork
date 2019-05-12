"""
This is an example of automatic differentiation
"""
import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.autograd import Variable
x = Variable( torch.tensor([.8]), requires_grad=True )
print(x)

def f(x):
    f = x**2
    return f

z = f(x)
#z = torch.tensor([z], requires_grad=True)
print("z=",z)
z.backward()
d = x.grad
print(d)
