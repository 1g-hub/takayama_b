# coding: utf-8
import time
from utils import nlp_util
import const
from itertools import chain

import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import transformers
import numpy as np

config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')
model = BertForMaskedLM.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin', config=config)
bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt', do_lower_case=False, do_basic_tokenize=False, config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')

s = ['[CLS]', 'この', 'アイドル', '事務', '所', 'で', 'は', '赤城', 'みりあ', 'を', '[MASK]', 'して', 'い', 'ます', '。', '[SEP]']

tokens = bert_tokenizer.convert_tokens_to_ids(s)
tokens_tensor = torch.tensor([tokens])
print(model.named_parameters)
masked_index = 10
optimizer = torch.optim.Adam(chain(model.parameters()), lr=0.00001)
print(tokens)
print(tokens_tensor)
model.eval()
with torch.no_grad():
 outputs = model(tokens_tensor)
 predictions = outputs[0]
 _, predicted_indexes = torch.topk(predictions[0, masked_index], k=10)
 predicted_tokens = bert_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist())
 print(predicted_tokens)