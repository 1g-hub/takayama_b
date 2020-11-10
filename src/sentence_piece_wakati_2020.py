# coding: utf-8
from collections import defaultdict

import numpy as np
import pandas as pd
import neologdn
import pickle
import os
from pandas.io.json import json_normalize

from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import sentencepiece as sp
TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
from collections import defaultdict

# sp_tokenizer = sp.SentencePieceProcessor()
# sp_tokenizer.load("../models/bert/hottoSNS-bert-pytorch/tokenizer_spm_32K.model")
#
# def create_wakati_sp(text):
#     text = text.lower()
#     text = (sp_tokenizer.EncodeAsPieces(text))
#     return ' '.join(text)
#
# for touch_name in TOUCH_NAME_ENG:
#     data = pd.read_csv(
#         '../new_dataset/' + touch_name + '_modified_aug.csv',
#         index_col=0,
#         header=0,
#         dtype={'id':int, 'original': bool,'inner': bool, 'kakimoji': bool, 'self_anotated': bool, 'alter_emotion':str},
#     )
#     print(touch_name)
#     # print(data.alter_emotion[0])
#     # if(np.isnan(data.alter_emotion[0])):
#     #     print("miria")
#     # #print(data)
#
#     for i, v in data.iterrows():
#
#         data.at[i, 'wakati_sp'] = create_wakati_sp(data.at[i, 'what'])
#         print(data.at[i, 'wakati_sp'])
#
#     data.to_csv('../new_dataset/' + touch_name + '_modified_aug.csv', encoding='utf_8_sig')

for touch_name in TOUCH_NAME_ENG:
    data = pd.read_csv(
        '../new_dataset/' + touch_name + '_modified_aug.csv',
        index_col=0,
        header=0,
        dtype={'id':int, 'original': bool,'inner': bool, 'kakimoji': bool, 'self_anotated': bool, 'alter_emotion':str},
    )
    print(data)
    data = data[data.kakimoji == False]
    d = defaultdict(int)
    for i, v in data.iterrows():
        d[data.at[i, 'emotion']] += 1
    print(touch_name)
    print(d)

for touch_name in TOUCH_NAME_ENG:
    data = pd.read_csv(
        '../new_dataset/' + touch_name + '_modified.csv',
        index_col=0,
        header=0,
        dtype={'id':int, 'original': bool,'inner': bool, 'kakimoji': bool, 'self_anotated': bool, 'alter_emotion':str},
    )
    data = data[data.kakimoji == False]
    d = defaultdict(int)
    for i, v in data.iterrows():
        d[data.at[i, 'emotion']] += 1
    print(touch_name)
    print(d)