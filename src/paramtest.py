from my_network.pytorch_mlp import ClassificationNet
import torch
from transformers import BertConfig, BertModel
import torch.nn as nn
c_net = ClassificationNet()
print(c_net)
for param in c_net.parameters():
    print(param)

config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')

bert_model = BertModel.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin', config=config)

print(bert_model)

for param in bert_model.parameters():
    print(param)
    break

class Net(nn.Module):
    def __init__(self, in_dim=768, out_dim=2):
        super(Net, self).__init__()
        self.emb = bert_model
        self.classifier = c_net

    def forward(self, x):
        x = self.emb(x)[0][:,0,:]

        return self.classifier(x)


n = Net()

#print(n)

load_weights = torch.load('../models/bert/My_Japanese_transformers/' + 'gyagu' + '_model.bin', map_location={'cuda:0': 'cpu'})
bert_model.load_state_dict(load_weights)

for param in bert_model.parameters():
    print(param)
    break