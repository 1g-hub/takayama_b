# coding: utf-8
import torchtext
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from torchtext.vocab import Vectors

# pytorch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np

class ClassificationNet(nn.Module):
    def __init__(self, in_dim=2, out_dim=5):
        super(ClassificationNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        return self.fc1(x)


def main():
    a = torch.tensor([[[2.0,3.0],[3.0,4.0],[5.0,6.0]], [[12.0,13.0],[13.0,14.0],[15.0,16.0]], [[22.0,32.0],[23.0,42.0],[25.0,62.0]], [[42.0,43.0],[53.0,54.0],[65.0,66.0]]])
    print(a.size())
    c = a.permute(1,0,2)
    net = ClassificationNet()
    print(c)

    c_out = net(c)
    print(c_out.size())


    out = c_out.permute(1,0,2)
    print(out.size())



    l = [[3,4],[4,8],[5,2]]
    hhh = np.stack([*l, [10,40]], axis=0)
    print(hhh)
    print(hhh.shape)

if __name__ == '__main__':
    main()