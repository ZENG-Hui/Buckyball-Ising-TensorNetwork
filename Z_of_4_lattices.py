import numpy as np
import torch
from scipy.linalg import sqrtm

beta = 1


# Tensor Network Method
B = np.array([ [np.exp(beta),  np.exp(-beta)],
                   [np.exp(-beta), np.exp(beta)] ])
B = torch.tensor(sqrtm(B))
A1 = B@B
A2 = A1
A3 = A1
A4 = A1

Z1 = torch.einsum('ij,jk->ik', A1, A2)
Z2 = torch.einsum('ij,jk->ik', A3, A4)
Z =  torch.einsum('ij,ij->', Z1, Z2)
Z = Z.numpy()
print('Z=',Z)

# Exact Result
ZZ = 2*np.exp(beta)*np.exp(beta)*np.exp(beta)*np.exp(beta) \
   + 2*np.exp(-beta)*np.exp(-beta)*np.exp(-beta)*np.exp(-beta) \
   + 12*np.exp(beta)*np.exp(-beta)*np.exp(beta)*np.exp(-beta) 

print('ZZ=',ZZ)

