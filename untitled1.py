# -*- coding: utf-8 -*-
"""
Created on Thu May  2 16:56:04 2019

@author: 46107
"""

import torch

x = torch.randn(2, 3)
torch.cat((x, x, x), 0)
torch.cat((x, x, x), 1)

x = torch.Tensor([[1], [2], [3]])
x = x.expand(3, 4)
x = x.expand(7,3,4)
# -1??

x = torch.zeros(2, 1, 2, 1, 2)
x.size()
#torch.Size([2, 1, 2, 1, 2])
y = torch.squeeze(x)
y.size()
#torch.Size([2, 2, 2])

for i in range(5):
    y = torch.squeeze(x, i)
    y.size()
#torch.Size([2, 2, 1, 2])
    
x = torch.Tensor([1, 2, 3])
x.repeat(4, 2)

x = torch.arange(1, 8)
x.unfold(0, 2, 1)
x.unfold(0, 2, 2)

x = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x.narrow(0, 0, 2)
x.narrow(1,1,2)

x = x.view([2,4])

x = torch.Tensor([[1, 2], [3, 4], [5, 6]])
x.resize_(4,5)
x.resize_(2,2)
x.resize_(2,3)

x.permute(1,0)