import numpy as np
import torch
from scipy import gradient
from scipy.linalg import sqrtm
from torch.autograd import Variable
from matplotlib import pyplot as plt


# define a function: compute the value of derivetive of function f(x)
# at x = a. Number of setps is N, and compute in the region [a-l, a+l]
def mydiff(f,a, N = 1e3, l = 1e-2):
    s = 2*l/N
    x = np.linspace(a-l,a+l,N)
    fun = np.linspace(0,0,N)
    for i in range(int(N)):
        fun[i] = f(x[i])
    d = gradient(fun,s)
    d = d[int(N)//2]
    return d

beta = 100


# compute the partition function Z
def getZ(beta,mu=0):
    B = np.array([ [np.exp(-beta), np.exp(beta)],
                       [np.exp(beta), np.exp(-beta)] ])

    
#    B = torch.tensor(sqrtm(B))
#    A1 = B@B
    B = B*np.exp(beta*mu)
    A1 = torch.tensor(B)

    A2 = A1
    A3 = A1

#    Z1 = torch.einsum('ij,jk->ik', A1, A2)
#    Z =  torch.einsum('ij,jk->ik', Z1, A3)
#    Z = torch.einsum("ii->",Z)
    Z = torch.einsum('jk,kl,lj,ij,ik,il->',A1,A1,A1,A1,A1,A1)
    Z = Z.numpy()
    return Z

print('Z=',getZ(beta))

# compute lnZ
def lnZ(beta):
    Z = getZ(beta)
    Z = np.log(Z)
    return Z

# when 1 is negleticable compare to beta, E is the ground state energy.
E0 = -mydiff(lnZ, beta, N = 1e3, l = 1e-4)
#E0 = np.around(E0)
print('E0 = ',E0)
#print('zzdaf', lnZ(.999))


mu = -2/6
print('mu=',mu)
# whe mu is equal to groud state energy E0/number of interaction
def Deg(beta,mu=mu):
    Deg = getZ(beta,mu=mu)
    return Deg

print('degeneracy is',Deg(beta))











#x = np.linspace(0,1,20)
#y = lnZ(x)
#plt.plot(x,y)
#plt.show()


#    print('Z=',Z)
#    print('ZZ=',ZZ)



'''
A1 = B@B
A2 = A1
A3 = A1
A4 = A1

Z1 = torch.einsum('ij,jk->ik', A1, A2)
Z2 = torch.einsum('ij,jk->ik', A3, A4)
Z =  torch.einsum('ij,jk->ik', Z1, Z2)
Z = torch.einsum('im->',Z)/2
print('Z=',Z)

ZZ = np.exp(beta)*np.exp(beta)*np.exp(beta)*np.exp(beta) \
   + np.exp(beta)*np.exp(beta)*np.exp(-beta)*np.exp(beta) \
   + np.exp(beta)*np.exp(-beta)*np.exp(beta)*np.exp(beta) \
   + np.exp(beta)*np.exp(-beta)*np.exp(-beta)*np.exp(beta) \
   + np.exp(-beta)*np.exp(beta)*np.exp(beta)*np.exp(beta) \
   + np.exp(-beta)*np.exp(beta)*np.exp(-beta)*np.exp(beta) \
   + np.exp(-beta)*np.exp(-beta)*np.exp(beta)*np.exp(beta) \
   + np.exp(-beta)*np.exp(-beta)*np.exp(-beta)*np.exp(beta) \
   + np.exp(beta)*np.exp(beta)*np.exp(beta)*np.exp(-beta) \
   + np.exp(beta)*np.exp(beta)*np.exp(-beta)*np.exp(-beta) \
   + np.exp(beta)*np.exp(-beta)*np.exp(beta)*np.exp(-beta) \
   + np.exp(beta)*np.exp(-beta)*np.exp(-beta)*np.exp(-beta) \
   + np.exp(-beta)*np.exp(beta)*np.exp(beta)*np.exp(-beta) \
   + np.exp(-beta)*np.exp(beta)*np.exp(-beta)*np.exp(-beta) \
   + np.exp(-beta)*np.exp(-beta)*np.exp(beta)*np.exp(-beta) \
   + np.exp(-beta)*np.exp(-beta)*np.exp(-beta)*np.exp(-beta) \
   

print('ZZ=',ZZ)
print(A1)
'''

print('-------------------')
A = torch.tensor([[1,2], [3,4]])
A = torch.tensor([[5,6], [7,8]])

