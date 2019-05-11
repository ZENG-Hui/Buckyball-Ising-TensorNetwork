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


beta = 1


def getZ(beta, mu = 0):
    S = np.array([ [np.exp(-beta), np.exp(beta)],\
                   [np.exp(beta), np.exp(-beta)]])
    S = S * np.exp(beta*mu)
    s = sqrtm(S)
    T= np.einsum('il,im,in->lmn',s,s,s)
    Z = np.einsum('ijk,iln,nmk,jlm->',T,T,T,T).real
    return Z
print('Tensor Network Z=',getZ(beta))


def getlogZ(beta):
    return np.log(getZ(beta))
E0 = - mydiff(getlogZ, 100, N=1e2, l = 1e-2)
print("Ground State Energy is", E0)

mu = -1
Deg = getZ(500, mu=mu)
print('Degeneracy is', Deg)
