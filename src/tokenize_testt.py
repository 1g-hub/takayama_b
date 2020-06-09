# coding: utf-8
from collections import defaultdict
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
def load_text(file_path):
    text = []
    with open(file_path, mode='r', encoding='utf_8') as f:
        for line in f:
            # 全角があれば半角に変更, 改行コードは取り除く
            line = line.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})).rstrip(
                '\n').replace('\u3000', ' ')
            if line != '':
                text.append(line)
    return text

file = '../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt'

bert_words = load_text(file)
print(bert_words)


# GLOBAL 変数
TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]


bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt', config='models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json', do_basic_tokenize=False)

test_s = 'A'

print(bert_tokenizer.convert_tokens_to_ids(test_s))

print(test_s in bert_words)


