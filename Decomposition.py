# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 20:36:33 2019

@author: 46107
"""
import numpy as np
import pandas
import torch
from numpy import linalg as LA
from numpy.linalg import matrix_rank

def reconstruct(W,sh,r,bits=1):
    W = W.numpy()
    print(np.shape(W))
    T = np.reshape(W,sh)
    #print(type(T))
    G,Case = TerTTSVD(T,0.01,r,2*np.ones([np.shape(r)[0]-1,1]),bits)
    
    if Case == False:
        print('Please reinput the rank')
        return
    error = LA.norm(ProTTSVD(G)-T)
    W  = ProTTSVD(G)
    return W,error,G

def TerTTSVD(A,eplison,r,bt,bits,Case=True):
# bt should be [dim-1] dimension vector of 1/2
    dim = np.shape(A)
    n = dim
    e = 0
    #delta = eplison*LA.norm(np.reshape(A,[dim[0],np.prod(dim)/dim[0]]),'fro')/np.sqrt(dim(n)[0]-1)
    C = A
    tC = C
    tr = np.zeros([len(n)+1])
    if (r[0] != 1 and r[-1]!=1):
        print('r(1)and r(end) should be 1')
        Case = False
        return Case
    tr[0] = 1
    tr[-1] = 1 
    G = [[] for i in range(np.shape(n)[0])]
    e = [[]]
    for k in range(np.shape(n)[0]-1):
        #print(np.shape(C))
        C = np.reshape(C,[int(tr[k]*n[k]), int(np.prod(n)/(tr[k]*n[k]))])
        tr[k+1] = matrix_rank(C)
        if r[k+1]> tr[k+1]:
            print('Input rank should be less than rank of tensor')
            Case = False
            return Case
        if bt[k] == 2:
            [M,a,R] = TerDecomMultibits(C,r[k+1], bits)
            
        elif bt[k] == 1:
            R = 10*randn(5,5)
            iter = 1
            while (norm(R)>0.01)and(iter < 2):
                [M,a,R] = BiDecomMultibits(C,r(k+1), bits)
                iter = iter + 1
        error = LA.norm(R,'fro')
        print('error is ',error)
        #e = e.append(error)
        Gk = np.reshape(M[:,:r[k+1]],[r[k],dim[k],r[k+1]])  
        
        G[k] = (Gk)
        #print(np.shape(G[k]))
        C = np.dot(LA.pinv(M),C)
        C = C[:r[k+1],:]
        tr[k+1] = r[k+1]
    G[-1] = C[:r[np.shape(n)[0]-1],:n[-1]].T

    return G, Case

def TerDecomMultibits(W,r,bits):
    R = W
    dim = np.shape(W)
    dim1 = dim[0]
    dim2 = dim[1]
    M = np.array(np.zeros([dim1,r]))
    a = np.array(np.zeros([r,dim2]))
    print('TerDecom starts!')
    for i in range(r):
        M[:,i] =np.sign(np.round(np.random.rand(dim1,)))#
        iter = 0
        
        while (iter<60):
            a[i,:] = (np.dot(M[:,i].T,R)/np.dot(M[:,i],M[:,i]))
            index = np.zeros(2**(bits)+1)
            
            for ii in range(-2**(bits-1),2**(bits-1)):
                index[ii+2**(bits-1)] = (ii)/2**(bits-1)
                #print('index',ii+2**(bits-1))
                #print('ii',ii)
            tempsum = np.zeros([len(index)])
            for j in range(dim1):
                for q in range(len(index)):
                    tempsum[q] = np.dot((R[j,:]-index[q]*a[i,:]),(R[j,:]-index[q]*a[i,:]))
                if np.isnan(np.min(tempsum)):
                    M[j,i] = index[1]
                else:
                    M[j,i] = index[np.nonzero(tempsum == min(tempsum))[0][0]]
            iter = iter + 1
        R = R - np.outer(M[:,i],a[i,:])
        print('iter {}/{} completed'.format(i+1,r))
    return M,a,R

def ProTTSVD(G):
    indexG = np.max(np.shape(G))
    n = [[] for i in range(indexG)]
    r = [[] for i in range(indexG+1)]
    r[indexG] = 1
    for i in range(indexG):
        #print(i)
        #print(np.shape(G[i]))
        temp = np.shape(G[i])
        n[i] = (temp[1])
        r[i] = (temp[0])
    temp = G[0]
    dim = np.shape(G[0])
    
    for i in range(1,indexG):
        #print(i)
        dim1 = [[] for _ in range(i+1+2)]
        temp = np.reshape(temp,[int(np.prod(np.shape(temp))/r[i]), int(r[i])])
        #print(np.shape(temp))
        temp = np.dot(temp,np.reshape(G[i],[r[i], int(np.prod(np.shape(G[i]))/r[i])]))
        #print(np.shape(temp))
        dim1[:i+1] = dim[:i+1]
        dim1[-2] = n[i]
        dim1[-1] = r[i+1]
        #print(dim1)
        temp = np.reshape(temp,dim1)
        dim = dim1
    #print(np.shape(temp))
    #print(dim)
    temp = np.reshape(temp,dim[1:-1])
    return temp


if __name__ == '__main__':
    main()
