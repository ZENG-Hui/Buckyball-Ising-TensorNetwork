"""
This is a python3.7 code to calculate the ground state energy of a
Ising model of C60 using tensor network method
copyright: Copyright 2018 by ZQW.
license: MIT
"""

import numpy as np
from scipy.linalg import sqrtm

def mydiff(f,a,b,N):
    """
    get the derivative of function
    f: the input function
    N: calculate the derivatives of function f at N points in [a, b]
    return: N derivatives of function f
    """
    fx = np.zeros(N)
    s = (b-a)/(N-1)
    d = np.zeros(N-1)
    for i in range(N):
        x = a+i*s
        fx[i] = f(x)
        print('f(',x,')=',fx[i])
        if i>0:
            d[i-1] = ( fx[i] - fx[i-1] ) /s
            print('f\' between',x,'and',x-s,'is',d[i-1])
    return d

def getlnZ(beta):
    S = np.array([ [np.exp(-beta), np.exp(beta)],\
                   [np.exp(beta), np.exp(-beta)]]) # the S matrix
    # between two lattice
    S = sqrtm(S)
    T = np.einsum('ij,ik,il->jkl',S,S,S) # the 3-tensor at each lattice
    T = np.einsum('iae,jab,kbc,lcd,mde->ijklm',T,T,T,T,T) # the
    # 5-tensor at each lattice of a icosahedron after contraction of 12 pentagon
    Z = np.einsum('abcde,kfejt,jdirs,ciqph,hongb,mgafl,\
                   ABCDz,Ayrqx,Bxpow,Cwnmv,Dvlku,zutsy->'\
                   ,T,T,T,T,T,T,T,T,T,T,T,T) # contract the 12
    # lattices of the icosahedron to get the partition function of C60
    Z = Z.real
    lnZ = np.log(Z)             # the log of partition of C60
    return lnZ

E0 = - mydiff( getlnZ, 1,12,12) # according to statistical mechanics
# to get the energy. when temperature is 0, it is the ground state energy.
print(E0)
