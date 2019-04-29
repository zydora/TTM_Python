# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 14:59:24 2019

@author: 46107
"""
import numpy as np

W = [1]
for i in range(1,16):
    W.append(i+1)
W = np.reshape(W,[4,4])
A = [1,2,3,4]