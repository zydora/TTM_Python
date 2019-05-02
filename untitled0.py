# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:10:57 2019

@author: 46107
"""


import Last_Update as L
import Decomposition as TT
import sys

def main():
    f = open('Jupyter.log', 'a')
    sys.stdout = f
    sys.stderr = f
    for b in range(4,6):
        L.update(b = b, i = 1)
        print(b,'_finished')

if __name__ == '__main__':
    main()
    