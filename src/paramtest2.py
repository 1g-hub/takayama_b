# coding: utf-8
from my_network.pytorch_mlp import ClassificationNet
import torch
from transformers import BertConfig, BertModel
import torch.nn as nn

TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
EPOCHS = 200 # エポック数
BATCH_SIZE = 16 # バッチサイズ


classifier = ClassificationNet()

criterion = torch.nn.CrossEntropyLoss()

w = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

class Net(nn.Module):
    config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')
    bert_premodel = BertModel.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin',
                                              config=config)

    def __init__(self):
        super(Net, self).__init__()
        self.bert = self.bert_premodel
        self.classifier = ClassificationNet()

    def save_params(self):
        pass

    def load_params(self):
        pass

    def forward(self, x):
        bert_out = self.bert(x)[0] # 最後の隠れ層
        out = self.classifier(bert_out[:, 0, :]) # [CLS] に相当する部分のみ使う
        return out

def save(net):
    torch.save(net.state_dict(), 'test_transformers_model.bin')

def load(net):
    load_weights = torch.load('test_transformers_model.bin',
                              map_location={'cuda:0': 'cpu'})
    net.load_state_dict(load_weights)


net = Net()
print(net.state_dict()['classifier.fc1.bias'])
print(net.state_dict()['bert.pooler.dense.bias'])
net.state_dict()['classifier.fc1.bias'][0] = 0.51415
net.state_dict()['bert.pooler.dense.bias'][0] = 2.82525
save(net)
load(net)
print(net.state_dict()['classifier.fc1.bias'])
print(net.state_dict()['bert.pooler.dense.bias'])