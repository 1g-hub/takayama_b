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

class ClassificationNet(nn.Module):
    def __init__(self, in_dim=768, out_dim=2):
        super(ClassificationNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, self.out_dim)

    def forward(self, x):
        return self.fc1(x)

class MLP3Net(nn.Module):
    def __init__(self, in_dim=300, hid_dim=30, out_dim=2):
        super(MLP3Net, self).__init__()
        # 3層MLP
        # 各隠れ層でのDropOut率は0.5
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.fc1 = nn.Linear(self.in_dim, self.hid_dim)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.hid_dim, self.out_dim)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x
