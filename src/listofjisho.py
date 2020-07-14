# coding: utf-8

import torch
a = {'1':[[2,3,4]],'2':[[3,4,5]]}

b = [c for c in a.items()]

print(b)

for i in range(6):
    # self.bert_encoder(x[i])[0][:, 0, :] 最後の隠れ層の先頭[CLS]に相当するベクトル
    token_type_ids = torch.tensor([[i % 2 for n in range(10)] for m in range(4)])
    print(token_type_ids)


from transformers import BertTokenizer, BertModel, BertConfig
bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt',
                                   do_lower_case=False, do_basic_tokenize=False,
                                   config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')
config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')
bert = BertModel.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin',
                                              config=config)

s = ["私", 'は', 'こども']

print(bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True))

d = bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)

for k,v in d.items():
    print(v)

n = [g for g in d['input_ids']]

print(n)
e = []

for i in range(len(d['input_ids'])):
    e.append({'input_ids': d['input_ids'][i],'token_type_ids': d['token_type_ids'][i], 'attention_mask': d['attention_mask'][i]})
#e = [{'input_ids': h['input_ids'],'token_type_ids': h['token_type_ids'], 'attention_mask': h['attention_mask']} for h in d]

print(e)

print(e[0]['input_ids'])


print("BERT input only")
print(bert(input_ids=torch.tensor(d['input_ids'])))
print(bert(input_ids=torch.tensor(d['input_ids']),token_type_ids=torch.tensor([[1,1,1],[1,1,1],[1,1,1]]),attention_mask=torch.tensor(d['attention_mask'])))
print("BERT")
# print(bert(**d))