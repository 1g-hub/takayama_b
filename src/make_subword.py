# coding: utf-8
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mojimoji

from transformers import BertTokenizer

bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt',
                                   do_lower_case=False, do_basic_tokenize=False,
                                   config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')

TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]

s = 'bさん、苦手そうだよな…'
print(s)
print(bert_tokenizer.tokenize(s))
print(bert_tokenizer.convert_tokens_to_ids(s))
print(bert_tokenizer.batch_encode_plus(s, pad_to_max_length=False, add_special_tokens=True, is_pretokenized=True))
print(bert_tokenizer.batch_encode_plus(s, pad_to_max_length=False, add_special_tokens=True, is_pretokenized=False))
s='b さん 、 苦手 そうだ よ な …'
print(s)
print(bert_tokenizer.tokenize(s))
print(bert_tokenizer.convert_tokens_to_ids(s))
print(bert_tokenizer.batch_encode_plus(s, pad_to_max_length=False, add_special_tokens=True, is_split_into_words=True))
print(bert_tokenizer.batch_encode_plus(s, pad_to_max_length=False, add_special_tokens=True, is_split_into_words=False))
for touch_name in TOUCH_NAME_ENG:
    data = pd.read_csv(
        '../new_dataset/' + touch_name + '_modified_aug.csv',
        index_col=0,
        header=0,
        dtype={'id': int, 'original': bool, 'inner': bool, 'kakimoji': bool, 'self_anotated': bool,
               'alter_emotion': str},
    )

    for index, row in data.iterrows():
        wakati = data.at[index, 'wakati'].split(" ")
        s_subword = []
        for w in wakati:
            if w == '!':
                w = '！'
            elif w == '?':
                w = '？'
            w = mojimoji.han_to_zen(w)
            word_after = bert_tokenizer.tokenize(w)
            for word in word_after:
                if word == '[UNK]':
                    word = ''.join(w)
                s_subword.append(word)
        data.at[index, 'subword'] = ' '.join(s_subword)
    data.to_csv('../new_dataset/' + touch_name + '_modified_aug_add_sub.csv', encoding='utf_8_sig')