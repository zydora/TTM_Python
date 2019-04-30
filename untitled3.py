# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:59:24 2019

@author: 46107
"""
import numpy as np
import tensorflow as tf
def main():
    W = [1]
    for i in range(1,16):
        W.append(i+1)
    print(W)
    W = np.reshape(W,[4,4])
    print(W)
    W = rreshape(W,[2,8])
    print(W)
    A = [1,2,3,4]

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
    #print(ssh)
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

if __name__ == '__main__':
    main()