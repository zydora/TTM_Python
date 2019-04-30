# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:10:57 2019

@author: 46107
"""


import Last_Update as L
import Decomposition as TT
import sys

def main():
    f = open('a.log', 'a')
    sys.stdout = f
    sys.stderr = f
    for bits in range(2,9):
        L.update(b = 0, i = bits)
        print(i,'_finished')

if __name__ == '__main__':
    main()
    