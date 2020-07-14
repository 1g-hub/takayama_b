# coding: utf-8
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from transformers import BertTokenizer

bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt',
                                   do_lower_case=False, do_basic_tokenize=False,
                                   config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')

TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]

for touch_name in TOUCH_NAME_ENG:
    data = pd.read_csv(
        '../dataset/' + touch_name + '_augmentation.csv',
        index_col=0,
        dtype={'original': bool},
        usecols=lambda x: x is not 'index'
    )

    for index, row in data.iterrows():
        wakati = data.at[index, 'wakati'].split(" ")
        s_subword = []
        for w in wakati:
            if w == '!':
                w = ''
            word_after = bert_tokenizer.tokenize(w)
            for word in word_after:
                if word == '[UNK]':
                    word = ''.join(w)
                s_subword.append(word)
        data.at[index, 'subword'] = ' '.join(s_subword)
    data.to_csv('../dataset/' + touch_name + '_aug_add_sub.csv', encoding='utf_8_sig')