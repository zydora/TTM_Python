# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 00:16:04 2019

@author: 46107
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
import numpy as np
import pandas
from numpy import linalg as LA
from numpy.linalg import matrix_rank

import Decomposition as TT
import train_and_save as TS

def update():
    # Decomposition para.
    reshape_size = [20,32,32,32]
    reshape_rank = [1,20,640,32,1]
    bitwidth = [1,2,3,4,8]
    bits = bitwidth[3]
    
    
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader  = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)


    #G = torch.load('G.pt')
    model = torch.load("mnist_cnn.pt")
    model = model()
    S = (model.fc1.weight)
    #print(S)
    S = S.detach()
    #print(S)
    
    [model.fc1.weight,error,G] = TT.reconstruct(S,reshape_size,reshape_rank,bits=bits)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        if epoch == args.epochs:
            np.savetxt('fc_weights',model.fc1.weight)
            np.savetxt('fc_bias',model.fc1.bias)

    if (args.save_model):
        #torch.save(model.state_dict(),"mnist_cnn.pt")
        pass
        
if __name__ == '__main__':
    update()