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

# その他もろもろ
import os
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 最後にattentionを可視化するときに使います。
import itertools
import random
from IPython.display import display, HTML

# nltkによる前処理用
import re
import nltk
from nltk import stem

class BiLSTMEncoder(nn.Module):
    def __init__(self, emb_dim=300, hidden_dim=128):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM (batch_size=1, 時系列データ数=2, 特徴量数=300 or 768)
        self.bilstm = nn.LSTM(input_size=emb_dim,
                              hidden_size=self.hidden_dim,
                              bidirectional=True,
                              batch_first=True,
                              bias=True
                             )
        # input_size: int -> 入力ベクトルの次元数
        # hidden_size: int -> 隠れ状態の次元数
        # *num_layers: int -> LSTMの層数。多層にしたいときは2以上に
        # *bias: bool -> バイアスを使うかどうか
        # *batch_first: bool
        # *dropout: float -> 途中の隠れ状態にDropoutを適用する確率
        # *bidirectional: bool -> 双方向LSTMにするかどうか


    def forward(self, x):
        # 各隠れ層のベクトルがほしいので第１戻り値を受け取る
        out, _ = self.bilstm(x)

        # x.shape => (batch, seq_len, emb_dim)

        # 前方向と後ろ方向の各隠れ層のベクトルを結合したままの状態で返す
        return out

class SelfAttention(nn.Module):
  def __init__(self, lstm_dim, da, r):
    super(SelfAttention, self).__init__()
    self.lstm_dim = lstm_dim
    self.da = da
    self.r = r
    self.main = nn.Sequential(
        # Bidirectionalなので各隠れ層のベクトルの次元は２倍のサイズになってます。
        nn.Linear(lstm_dim * 2, da),
        nn.Tanh(),
        nn.Linear(da, r)
    )
  def forward(self, out):
    return F.softmax(self.main(out), dim=1)


class SelfAttentionClassifier(nn.Module):
  def __init__(self, lstm_dim, da, r, tagset_size):
    super(SelfAttentionClassifier, self).__init__()
    self.lstm_dim = lstm_dim
    self.r = r
    self.attn = SelfAttention(lstm_dim, da, r)
    self.main = nn.Linear(lstm_dim * 6, tagset_size)

  def forward(self, out):
    attention_weight = self.attn(out)
    m1 = (out * attention_weight[:,:,0].unsqueeze(2)).sum(dim=1)
    m2 = (out * attention_weight[:,:,1].unsqueeze(2)).sum(dim=1)
    m3 = (out * attention_weight[:,:,2].unsqueeze(2)).sum(dim=1)
    feats = torch.cat([m1, m2, m3], dim=1)
    # F.log_softmax(self.main(feats), dim=1)
    output = self.main(feats)
    print(output.size())
    print(output)
    return output, attention_weight