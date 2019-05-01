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
import tensorflow as tf
import random
import scipy.io as sio

def reconstruct(W,sh,r,itera,bits=2):
    W = W.numpy()
    T = rreshape(W,sh)
    G,Case = TerTTSVD(T,0.01,r,2*np.ones([np.shape(r)[0]-1,1]),bits,itera,Case=True)
    print('itera is:',itera)
    if Case == False:
        print('Please reinput the rank')
        return
    #torch.save(G, "G"+str(bits)+".pt")
    #for i in range(3):
    #    print(np.shape(G[i]))
    W  = ProTTSVD(G[:])
    #print(np.shape(W))
    error = LA.norm(W-T)
    print('reconstruction accomplished!')
    return W,error,G

def TerTTSVD(A,eplison,r,bt,bits,itera,Case=True):
# bt should be [dim-1] dimension vector of 1/2
    dim = np.shape(A)
    n = dim
    e = 0
    #delta = eplison*LA.norm(rreshape(A,[dim[0],np.prod(dim)/dim[0]]),'fro')/np.sqrt(dim(n)[0]-1)
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
        C = rreshape(C,[int(tr[k]*n[k]), int(np.prod(np.shape(C))/(tr[k]*n[k]))])
        tr[k+1] = matrix_rank(C)
        #print(tr[k+1])
        if r[k+1]> tr[k+1]:
            print('Input rank should be less than rank of tensor')
            Case = False
            return Case
        if bt[k] == 2:
            [M,a,R] = TerDecomMultibits(C,r[k+1],itera, bits,k+1)
            #print(M)
            '''
        elif bt[k] == 1:
            print('error)=')
            R = 10*randn(5,5)
            iter = 1
            while (norm(R)>0.01)and(iter < 2):
                [M,a,R] = BiDecomMultibits(C,r(k+1), bits)
                iter = iter + 1
                '''
        error = LA.norm(R,'fro')
        print('error is ',error)
        #e = e.append(error)
        G[k] = rreshape(M[:,:r[k+1]],[r[k],dim[k],r[k+1]])
        C = np.dot(LA.pinv(M),C)
        #print(LA.pinv(M))
        #print(C)
        C = C[:r[k+1],:]
        tr[k+1] = r[k+1]
    temp = C[:r[np.shape(n)[0]-1],:n[-1]]
    p_arr = np.concatenate((np.shape(temp),[1]))
    G[-1] = rreshape(temp,p_arr)
    #print('itera is:',itera)
    return G, Case

def TerDecomMultibits(W,r,itera,bits,InitM):
    R = W
    dim = np.shape(W)
    dim1 = dim[0]
    dim2 = dim[1]
    M = np.array(np.zeros([dim1,r]))
    a = np.array(np.zeros([r,dim2]))
    print('TerDecom starts!')
    for i in range(r):
        M[:,i] =np.sign(np.round(np.random.randn(dim1,)))
        iter = 0
        while (iter<itera):
            a[i,:] = (np.dot(M[:,i].T,R)/np.dot(M[:,i].T,M[:,i]))
            index = np.zeros(2**(bits)+1)
            for ii in range(-2**(bits-1),2**(bits-1)+1):
                index[ii+2**(bits-1)] = (ii)/2**(bits-1)
            tempsum = np.zeros([len(index)])
            for j in range(dim1):
                for q in range(len(index)):
                    tempsum[q] = np.dot((R[j,:]-index[q]*a[i,:]),(R[j,:]-index[q]*a[i,:]).T)
                if np.isnan(np.min(tempsum)):
                    M[j,i] = index[1]
                else:
                    M[j,i] = index[np.nonzero(tempsum == min(tempsum))[0][0]]
            iter = iter + 1
        R = R - np.outer(M[:,i],a[i,:])
        print('iter {}/{} completed'.format(i+1,r))
    #print('itera is:',itera)
    return M,a,R

def ProTTSVD(G):
    indexG = np.max(np.shape(G))
    n = [[] for i in range(indexG)]
    r = [[] for i in range(indexG+1)]
    
    for i in range(indexG):
        temp = np.shape(G[i])
        n[i] = (temp[1])
        r[i] = (temp[0])
    
    r[indexG] = np.shape(G[-1])[-1]
    temp = G[0]
    dim = np.shape(G[0])
    
    for i in range(1,indexG):
        dim1 = [[] for _ in range(i+1+2)]
        temp = rreshape(temp,[int(np.prod(np.shape(temp))/r[i]), int(r[i])])
        temp = np.dot(temp,rreshape(G[i],[r[i], int(np.prod(np.shape(G[i]))/r[i])]))
        dim1[:i+1] = dim[:i+1]
        dim1[-2] = n[i]
        dim1[-1] = r[i+1]
        temp = rreshape(temp,dim1)
        dim = dim1
    if (dim[-1] == 1) and (dim[0] == 1):
        ###########
        # TT
        ###########
        temp = rreshape(temp,dim[1:-1])
    elif dim[-1] != 1:
        ###########
        # TTM/MPO
        ###########
        temp = rreshape(temp,dim[1:])
    print('PROTTSVD accomplished!')
    return temp

def rreshape(W,sh):
    '''
    print(sh)
    print('aaaaaaaaaaaaaaaaa')
    print(np.shape(W))
    '''
    # 1
    W = trans01(W)
    # 2
    ssh = np.array(sh)
    a = sh[0]
    b = sh[1]
    ssh[1] = a
    ssh[0] = b
    W = np.reshape(W,ssh)
    # 3
    W = trans01(W)
    '''
    print('bbbbbbbbbbbbbbbbbbbbb')
    print(np.shape(W))
    '''
    return W

def trans01(W):
    LL = [[] for i in range(np.shape(np.shape(W))[0])]
    for i in range(np.shape(np.shape(W))[0]):
        LL[i] = i
    a = LL[0]
    LL[0] = LL[1]
    LL[1] = a#np.shape(np.shape(W))[0]-1
    W = tf.transpose(W, perm=LL)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        W = (sess.run(W))
    return W

'''
def load(M,k):
    data = sio.loadmat('InitialM_'+str(k)+'_.mat')
    tM = data['tM']
    return tM
    '''
'''
def main():
    W = np.random.rand(100,60)
    W = torch.Tensor(W)
    [W,error,G] = reconstruct(W,[5,5,12,20],[1,5,25,20,1],bits=1,itera=20)
    '''
if __name__ == '__main__':
    main()
