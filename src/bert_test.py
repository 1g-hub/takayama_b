import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

s = [["すもも", "も", "もも", "も", "もも", "の", "うち"], ["みりあ", "は", "可愛い"]]


encoded_data = bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)

print(encoded_data)

input_ids = torch.tensor(encoded_data["input_ids"])
print(bert_tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))
print(bert_tokenizer.convert_ids_to_tokens(input_ids[1].tolist()))