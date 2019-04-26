# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 17:51:49 2019

@author: 46107
"""

import torch
import torchvision

class MnistDataLoaderWrapper(object):
    def _init_(self):
        batch_size = 32
        self.train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('../data', train=True, download=True,
                       transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                batch_size=batch_size, shuffle=False, **kwargs)
        self.test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('../data', train=False, download=True,
                       transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor(),
                           torchvision.transforms.Normalize((0.1307,), (0.3081,))
                       ])),
                batch_size=batch_size, shuffle=False, **kwargs)