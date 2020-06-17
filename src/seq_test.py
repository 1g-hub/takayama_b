# coding: utf-8
from manga4koma import manga4koma
from utils.history import History
from my_network.pytorch_mlp import ClassificationNet, MLP3Net
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
from my_network import pytorch_self_attention as self_net
from torchsummary import summary


class Net(nn.Module):
    config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')
    def __init__(self, seq_len=3, fine_tuning=True):
        super(Net, self).__init__()
        self.bert_encoder = BertModel.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin',
                                              config=self.config)
        self.bi_lstm = self_net.BiLSTMEncoder(768, 128)
        self.classifier = self_net.SelfAttentionClassifier(128, 64, 3, 2)

        self.seq_len = seq_len
        self.fine_tuning = fine_tuning

        # Bertの1〜11段目は更新せず、12段目とSequenceClassificationのLayerのみトレーニングする。
        # 一旦全部のパラメータのrequires_gradをFalseで更新
        for name, param in self.bert_encoder.named_parameters():
            param.requires_grad = False

        if self.fine_tuning:
            # Bert encoderの最終レイヤのrequires_gradをTrueで更新
            for name, param in self.bert_encoder.encoder.layer[-1].named_parameters():
                param.requires_grad = True

    def forward(self, x):
        x = x.type(torch.long)
        print("input shape : {}".format(x.size()))
        batch_size = list(x.size())[0]
        x = x.permute(1, 0, 2)
        print("before bert shape : {}".format(x.size()))
        bert_out = torch.empty(self.seq_len,batch_size,768)
        for i in range(self.seq_len):

            # if i == 0:
            #     bert_out = self.bert_encoder(x[i])[0][:, 0, :]

            bert_out[i] = self.bert_encoder(x[i])[0][:, 0, :]  # self.bert_encoder(x[i])[0][:, 0, :] 最後の隠れ層の先頭[CLS]に相当するベクトル
            print("bert out[i] shape : {}".format(bert_out[i].size()))

        bert_out = bert_out.permute(1, 0, 2)
        print("bert out shape(permutated) : {}".format(bert_out.size()))
        bi_lstm_out = self.bi_lstm(bert_out)
        print("bi_lstm_out shape : {}".format(bi_lstm_out.size()))
        out, attn = self.classifier(bi_lstm_out)
        print("out shape : {}".format(out.size()))
        return out


def main():
    a = torch.tensor([[[[2.0, 3.0], [3.0, 4.0], [5.0, 6.0]], [[12.0, 13.0], [13.0, 14.0], [15.0, 16.0]], [[22.0, 32.0], [23.0, 42.0], [25.0, 62.0]], [[42.0, 43.0], [53.0, 54.0], [65.0, 66.0]]],
                      [[[12.0, 13.0], [3.0, 14.0], [15.0, 16.0]], [[22.0, 33.0], [53.0, 4.0], [25.0, 6.0]], [[42.0, 12.0], [64.0, 24.0], [52.0, 17.0]], [[32.0, 12.0], [32.0, 11.0], [78.0, 34.0]]]]).long()
    b = torch.tensor([[[2.0, 3.0], [3.0, 4.0], [5.0, 6.0]], [[12.0, 13.0], [13.0, 14.0], [15.0, 16.0]], [[22.0, 32.0], [23.0, 42.0], [25.0, 62.0]], [[42.0, 43.0], [53.0, 54.0], [65.0, 66.0]]]).long()
    target = torch.tensor([[0,1],[1,0],[1,0],[0,1]]).long()

    net = Net(seq_len=3,fine_tuning=False)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.00001)
    criterion = torch.nn.CrossEntropyLoss()
    out = net(b)
    print(out)
    loss = criterion(out, target.argmax(1))
    loss.backward()
    optimizer.step()

    summary(net, (3, 32), batch_size=4, device='cpu')


if __name__ == '__main__':
    main()