import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

s = [["すもも", "も", "もも", "も", "もも", "の", "うち"], ["みりあ", "は", "可愛い"]]
config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')
model = BertForMaskedLM.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin', config=config)
bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt', do_lower_case=False, do_basic_tokenize=False, config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')


encoded_data = bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)

print(encoded_data)

input_ids = torch.tensor(encoded_data["input_ids"])
print(bert_tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
print(bert_tokenizer.convert_ids_to_tokens(input_ids[1].tolist()))