# coding: utf-8
from manga4koma import manga4koma
from utils.history import History
from my_network.pytorch_mlp import ClassificationNet, MLP3Net
from my_network import pytorch_self_attention as self_net
import torch
from transformers import BertConfig, BertModel
import matplotlib.pyplot as plt
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ========
# 最終層のみのfine-tuning
class Net(nn.Module):
    #config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')
    def __init__(self, seq_len=3, fine_tuning=True):
        super(Net, self).__init__()

        if manga_data.mode == 'kyoto':
            self.config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')
            self.bert_encoder = BertModel.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin',
                                              config=self.config)
        else:
            self.config = BertConfig.from_json_file('../models/bert/hottoSNS-bert-pytorch/config.json')
            self.bert_encoder = BertModel.from_pretrained(
                '../models/bert/hottoSNS-bert-pytorch/hottoSNS-bert-pytorch.bin',
                config=self.config)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None):

        bert_out = self.bert_encoder(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]  # 最後の隠れ層
        return bert_out[:, 0, :]

# ベクトル同士のコサイン類似度
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)  # 正規化
    ny = y / (np.sqrt(np.sum(y**2)) + eps)  # 正規化
    return np.dot(nx, ny)


# hotto

manga_data = manga4koma(to_zero_pad=False, to_sequential=False, seq_len=3, mode='hotto')
data = manga_data.make_5touch_concat()
ori_data = data[(data.original) & (data.touch == 'gyagu') & (data.story_main_num < 5)]
net = Net().to(device)


res_hotto =[]
cnt = 0
for index, ori in ori_data.iterrows():
    aug_data = data[(data.id == ori.id) & (data.touch == ori.touch) & (data.original is not True)]
    # print(ori.what)
    # print(len(aug_data))
    for i, aug in aug_data.iterrows():
        cnt +=1
        # aug
        x_input_ids = Variable(torch.tensor([aug.input_ids])).to(device)
        x_tok = Variable(torch.tensor([aug.token_type_ids])).to(device)
        x_attn = Variable(torch.tensor([aug.attention_mask])).to(device)

        #original
        o_input_ids = Variable(torch.tensor([ori.input_ids])).to(device)
        o_tok = Variable(torch.tensor([ori.token_type_ids])).to(device)
        o_attn = Variable(torch.tensor([ori.attention_mask])).to(device)

        net.eval()
        with torch.no_grad():
            x = net(input_ids=x_input_ids, token_type_ids=x_tok, attention_mask=x_attn)[0].to('cpu').detach().numpy().copy()
            o = net(input_ids=o_input_ids, token_type_ids=o_tok, attention_mask=o_attn)[0].to('cpu').detach().numpy().copy()
            res_hotto.append(cos_similarity(x, o))
print(cnt)
# kyoto

print("####################################\n\nKYOTO\n\n################################")

manga_data = manga4koma(to_zero_pad=False, to_sequential=False, seq_len=3, mode='kyoto')
data = manga_data.make_5touch_concat()
ori_data = data[(data.original) & (data.touch == 'gyagu') & (data.story_main_num < 5)]
net = Net().to(device)


res_kyoto =[]
cnt = 0
for index, ori in ori_data.iterrows():
    aug_data = data[(data.id == ori.id) & (data.touch == ori.touch) & (data.original is not True)]
    # print(ori.what)
    # print(len(aug_data))
    for i, aug in aug_data.iterrows():
        cnt+=1
        # aug
        x_input_ids = Variable(torch.tensor([aug.input_ids])).to(device)
        x_tok = Variable(torch.tensor([aug.token_type_ids])).to(device)
        x_attn = Variable(torch.tensor([aug.attention_mask])).to(device)

        #original
        o_input_ids = Variable(torch.tensor([ori.input_ids])).to(device)
        o_tok = Variable(torch.tensor([ori.token_type_ids])).to(device)
        o_attn = Variable(torch.tensor([ori.attention_mask])).to(device)

        net.eval()
        with torch.no_grad():
            x = net(input_ids=x_input_ids, token_type_ids=x_tok, attention_mask=x_attn)[0].to('cpu').detach().numpy().copy()
            o = net(input_ids=o_input_ids, token_type_ids=o_tok, attention_mask=o_attn)[0].to('cpu').detach().numpy().copy()
            res_kyoto.append(cos_similarity(x, o))

print(cnt)
bins = np.linspace(-1,1,50)
plt.hist(res_hotto, bins, alpha = 0.5, label='hotto-SNS')
plt.hist(res_kyoto, bins, alpha = 0.5, label='Kyoto')
plt.xlabel("cos_similarity")
plt.ylabel("num_data")
plt.legend(loc='upper left')
plt.savefig("./cos_sim_gyagu_hotto_vs_kyoto.png")
plt.clf()