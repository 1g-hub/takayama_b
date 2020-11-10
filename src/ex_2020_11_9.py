# coding: utf-8
from manga4koma import manga4koma
from utils.history import History
from my_network.pytorch_mlp import ClassificationNet, MLP3Net
from my_network import pytorch_self_attention as self_net
import torch
from transformers import BertConfig, BertModel
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import optuna
from sklearn.metrics import classification_report
import numpy as np
from utils.visualizer import Visualizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn

# ========
# GLOBAL 変数
TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]

P_EMOTIONS = ['喜楽']

P_DIC = {'ニュートラル':'neutral', '驚愕':'kyougaku', '喜楽':'kiraku'}

manga_data = manga4koma(to_zero_pad=True, to_sequential=True, seq_len=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ========