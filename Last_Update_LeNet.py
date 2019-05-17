# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 00:16:04 2019

@author: 46107
"""
from __future__ import print_function
import argparse
import torch
import torchvision
#mport torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from DataLoader import MnistDataLoaderWrapper
import Decomposition as TT
from train_and_save_Lenet import Net
from numpy import linalg as LA
import tensorflow as tf
import scipy.io as sio

import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import pandas as pd
import sys

def update(b = 0, i = 1, m = 0, s = 0):
    reshape_size_1 = [[20,32,32,32],[640,1024]]
    reshape_size = reshape_size_1[0]
    reshape_rank = [1,20,640,32,1]
    #reshape_rank = [1,640,1]
    bitwidth = [1,2,3,4,6,8]
    bits = bitwidth[b]
    iteration = [20,60,100,200]
    itera = iteration[i]
    step_size = [0.000001,0.00001, 0.0001]
    step = step_size[s]
    print('stepsizeis',step)
    RTrain = True
    decom = False
    Modell = ["LeNet-5","VGG16","VGGFace"]
    modell = Modell[m]
    print('Starts!')
    print('---------------------bits',bits,'--------------------')
    print('---------------------itera',itera,'--------------------')
####################################################
# Training settings                                 
####################################################
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
                                   #transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
    batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                           transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   #transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)
    if modell == "LeNet-5":
        model = torch.load("mnist_cnn.pt")
    elif modell == "VGG16":
        model = models.vgg16(pretrained=True)
        print(model.classifier[0].weight)
    elif modell == "VGGFace":
        model = torch.load("VGGFace.pt")
####################################################
# Decomposition and Reconstruction-LeNet
####################################################
    if decom == True:
        [W,error,G] = TT.reconstruct(model.fc1.weight.data,reshape_size,reshape_rank,itera=itera,bits=bits)
        print(LA.norm(error))
        torch.save(G, "G_"+modell+str(bits)+".pt")
    else:
        print('Don\'t decom, just load')
    G = torch.load("G_"+modell+str(bits)+".pt")
    A1 = TT.ProTTSVD(G[:-1])
    A1 = rreshape(A1,[np.prod(np.shape(A1)[:-1]),np.shape(A1)[-1]])#[n1n2n3, r4]
    W = rreshape((TT.ProTTSVD(G)),np.shape(model.fc1.weight.data))
    error = torch.Tensor(W)-model.fc1.weight.data
    print(LA.norm(error))
    if LA.norm(error)!=0:
        print('model substitude')
    model.fc1.weight.data = torch.Tensor(W)
#####################################################
# train and test
#####################################################
    
    '''
    data_loader = MnistDataLoaderWrapper()
    train_loader = data_loader.train_loader
    test_loader = data_loader.test_loader
    '''
    if RTrain == True:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        print('--------------------------')
        print('Now updating the last core')
        temp = model.fc1.weight.data
        test(args, model, device, test_loader)
        train(args, model, device, test_loader, optimizer, step,A1, G, reshape_size, epoch=1, Update_last_core = True)
        #print('--------------', model.fc1.weight.data-temp)
        test(args, model, device, test_loader)
    elif RTrain == False:
        print('Without train, just test reconstruction accuracy')
        print('Train set')
        test(args, model, device, train_loader)
        print('Test set')
        test(args, model, device, test_loader)
    
def train(args, model, device, train_loader, optimizer, step,A1, G, reshape_size, epoch, Update_last_core = False):
    model.train()
#####################################################
# freeze the param.
#####################################################
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc1.parameters():
        param.requires_grad = True
    print('Loss_grad is in grabbing progression')
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)
        optimizer.zero_grad()
        if model.fc1.weight.requires_grad == True or model.fc2.weight.requires_grad == True:
            loss.backward()
##########################################
# Update last core
##########################################
        if model.fc1.weight.requires_grad == True and Update_last_core == True:
            Loss_grad = model.fc1.weight.grad
            Q = rreshape(Loss_grad,[np.prod(reshape_size[:-1]),reshape_size[-1]])#[n1n2n3,n4]
            #print(A1)
            print('=========================')
            Size_temp = np.append((np.shape(np.dot(A1.T,Q))),1)#[r4,n4,r5=1]
            
            #tempG = G
            #tG = tempG
            G[-1] = G[-1]- step*rreshape(np.dot(A1.T,Q),Size_temp)
            '''
            print(np.dot(A1.T,Q))
            print('-----------------')
            #print(np.dot(A1,tG[-1]))
            print(np.dot(A1,np.dot(A1.T,Q)))
            print('+++++++++++++++')
            print(G[0])
            print(G[1])
            print(G[2])
            print(G[-1])
            '''
            W = rreshape((TT.ProTTSVD(G)),np.shape(model.fc1.weight.data))
            error = torch.Tensor(W)-model.fc1.weight.data
            print(error)
            print('Updating error',LA.norm(error))
            model.fc1.weight.data = torch.Tensor(W)
            
###########################################
# Initialize the grad
###########################################
            model.fc1.weight.grad = torch.Tensor(np.zeros(np.shape(model.fc1.weight.grad)))
            model.fc1.bias.grad = torch.Tensor(np.zeros(np.shape(model.fc1.bias.grad)))
            
        elif model.fc1.weight.requires_grad == True and Update_last_core == False:
            print('We do not update the last core here')
        '''
        for param in model.fc1.parameters():
            param.requires_grad = False
        for param in model.fc2.parameters():
            param.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        '''
###########################################
# Train
###########################################
        #optimizer.step()
########### test #######################
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        #return
    
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def rreshape(W,sh):
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
    return W

def trans01(W):
    LL = [[] for i in range(np.shape(np.shape(W))[0])]
    for i in range(np.shape(np.shape(W))[0]):
        LL[i] = i
    a = LL[0]
    LL[0] = LL[1]
    LL[1] = a
    W = tf.transpose(W, perm=LL)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        W = (sess.run(W))
    return W

if __name__ == '__main__':
    np.random.seed(5)
    '''
    f = open('0512.log', 'a')
    sys.stdout = f
    sys.stderr = f
    '''
    for s in range(3):
        update(b = 0, i = 1, m = 0, s = 0)
        print(s,'_finished')
