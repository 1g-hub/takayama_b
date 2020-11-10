# coding: utf-8
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertModel
import sentencepiece as sp
#
# config = BertConfig.from_json_file('../models/bert/hottoSNS-bert-pytorch/config.json')
# model = BertForMaskedLM.from_pretrained('../models/bert/hottoSNS-bert-pytorch/hottoSNS-bert-pytorch.bin',
#                                               config='../models/bert/hottoSNS-bert-pytorch/config.json')
bert_tokenizer = BertTokenizer('../models/bert/hottoSNS-bert-pytorch/vocab.txt',config='../models/bert/hottoSNS-bert-pytorch/tokenizer_config_modified.json')
# オリジナルの文章
s = ['[CLS]', '隣', 'の', '客', 'は', 'よく', '柿', '食う', '客', 'だ', '[SEP]']
s = 'Bさん、苦手そうだよな…'
s_predicts = []
print(bert_tokenizer.tokenize('B'))
print(bert_tokenizer.tokenize('B'))
print(bert_tokenizer.tokenize(s,is_split_into_words=False))
print(bert_tokenizer.tokenize(s,is_split_into_words=True))
sp_tokenizer = sp.SentencePieceProcessor()
sp_tokenizer.load("../models/bert/hottoSNS-bert-pytorch/tokenizer_spm_32K.model")
s = ' '.join(sp_tokenizer.EncodeAsPieces(s))
print(s)
print(bert_tokenizer.tokenize(s,is_split_into_words=False))
print(bert_tokenizer.tokenize(s,is_split_into_words=True))
s = bert_tokenizer.tokenize(s,is_split_into_words=True)
tokens = bert_tokenizer.convert_tokens_to_ids(s) # 単語 id 列に変換
print(tokens)
tokens_tensor = torch.tensor([tokens]) # 入力用にテンソルにする
print(tokens) # [2, 3893, 5, 1262, 9, 886, 14049, 1, 1262, 234, 3]
print(tokens_tensor) # torch.tensor([2, 3893, 5, 1262, 9, 886, 14049, 1, 1262, 234, 3])

# with torch.no_grad():
#     # [CLS], [SEP] 以外について推測させる
#     for masked_index in range(0, len(s)):
#         s_s = s.copy()
#         s_s[masked_index] = "[MASK]" # [MASK]
#         tokens = bert_tokenizer.convert_tokens_to_ids(s_s)  # 単語 id 列に変換
#         tokens_tensor = torch.tensor([tokens])  # 入力用にテンソルにする
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
# for e in s_predicts:
#     print(e)