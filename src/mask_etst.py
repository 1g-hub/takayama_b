# coding: utf-8

# EXAMPLE
# BertForMaskedLM を用いて マスク問題を解く.

import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')
model = BertForMaskedLM.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin', config=config)
bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt', config='models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')

# オリジナルの文章
s = ['[CLS]', '女の子', '同士', 'の', '恋愛', 'を', '[MASK]', 'と', 'いう', '。', '[SEP]']
s2 = ['[CLS]', '安達', 'と', 'しまむら', 'は', '[MASK]', 'で', 'ある', '。', '[SEP]']
s_predicts = []

tokens = bert_tokenizer.convert_tokens_to_ids(s) # 単語 id 列に変換
tokens_tensor = torch.tensor([tokens]) # 入力用にテンソルにする
print(tokens) # [2, 3893, 5, 1262, 9, 886, 14049, 1, 1262, 234, 3]
print(tokens_tensor) # torch.tensor([2, 3893, 5, 1262, 9, 886, 14049, 1, 1262, 234, 3])

s3 = ['!', '！']
tokens = bert_tokenizer.convert_tokens_to_ids(s3) # 単語 id 列に変換
print(tokens)

# with torch.no_grad():
#     # [CLS], [SEP] 以外について推測させる
#     for masked_index in range(1, len(s) - 1):
#         if masked_index != 5:
#             continue
#         outputs = model(tokens_tensor)
#         predictions = outputs[0] # 最終の隠れ層を用いる
#         _, predicted_indexes = torch.topk(predictions[0, masked_index], k=10) # 推測結果の上位 k 個の単語 id を取得
#         predicted_tokens = bert_tokenizer.convert_ids_to_tokens(predicted_indexes.tolist()) # 単語 id 列 -> 単語列
#
#         # 推測結果ごとに単語を置換して追加
#         for token in predicted_tokens:
#             tmp = s.copy()
#             tmp[masked_index] = token
#             s_predicts.append(tmp)
#
# for predict in s_predicts:
#     print(predict)