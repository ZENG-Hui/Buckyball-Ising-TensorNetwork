import numpy as np
import torch
from scipy.linalg import sqrtm
from torch.autograd import Variable
x = Variable( torch.tensor([.8]), requires_grad=True )
print(x)

def f(x):
    f = x.exp()
    return f

z = f(x)
#z = torch.tensor([z], requires_grad=True)
print("z=",z)
z.backward()
d = x.grad
print(d)
