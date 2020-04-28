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

