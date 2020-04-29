# coding: utf-8
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import nlp_util as nlp

import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig

from manga4koma_embedding import Embbedding

TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
CONV_TOUCH = {"gyagu": "ギャグタッチ", "shoujo": "少女漫画タッチ", "shounen": "少年漫画タッチ", "seinen": "青年漫画タッチ", "moe": "萌え系タッチ"}
EMB_MODE = ["d2v", "bert"]
EMB_DIM = {"d2v": 300, "bert": 768}
CONV_EMO = {"ニュートラル": 'neutral', "驚愕": 'kyougaku', "喜楽": 'kiraku', "恐怖": 'kyouhu', "悲哀": 'hiai', "憤怒": 'hunnu', "嫌悪": 'keno'}
SEQ_LEN = [2, 3, 4, 5, 6]

bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt',
                                   do_lower_case=False, do_basic_tokenize=False,
                                   config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')

class manga4koma():
    def __init__(self, to_zero_pad=False):
        self.TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
        self.CONV_TOUCH = {"gyagu": "ギャグタッチ", "shoujo": "少女漫画タッチ", "shounen": "少年漫画タッチ", "seinen": "青年漫画タッチ", "moe": "萌え系タッチ"}
        self.EMB_MODE = ["d2v", "bert"]
        self.EMB_DIM = {"d2v": 300, "bert": 768}
        self.CONV_EMO = {"ニュートラル": 'neutral', "驚愕": 'kyougaku', "喜楽": 'kiraku', "恐怖": 'kyouhu', "悲哀": 'hiai', "憤怒": 'hunnu', "嫌悪": 'keno'}
        self.SEQ_LEN = [2, 3, 4, 5, 6]
        self.to_zero_pad = to_zero_pad
        self.__set_data()
        self.__tokenize()

    def __set_data(self):
        self.data = defaultdict(pd.DataFrame)
        self.original_data = defaultdict(pd.DataFrame)
        self.bert_tokenized = defaultdict(dict)
        self.tokenized = defaultdict(list)

        for touch_name in self.TOUCH_NAME_ENG:
            self.data[touch_name] = pd.read_csv(
                '../dataset/' + touch_name + '_augmentation.csv',
                index_col=0,
                dtype={'original': bool},
                usecols=lambda x: x is not 'index'
            )
            self.original_data[touch_name] = self.data[touch_name][self.data[touch_name].original]

            self.data[touch_name].wakati = [w.split(' ') for w in self.data[touch_name].wakati.tolist()]
            self.original_data[touch_name].wakati = [w.split(' ') for w in self.original_data[touch_name].wakati.tolist()]

    def __tokenize(self, seq_len=3):
        for touch_name in self.TOUCH_NAME_ENG:
            if self.to_zero_pad:
                self.zero_padding(touch_name, seq_len)
            s = self.data[touch_name].wakati.tolist()
            #self.new_data[touch_name] = self.data[touch_name].assign(to)
            self.data[touch_name]['tokenized'] = s
            self.data[touch_name]['bert_tokenized'] = [b for b in bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)['input_ids']]


    def zero_padding(self, touch_name, seq_len=3):
        data_columns = ['id', 'original', 'story_main_num', 'story_sub_num', 'koma', 'who', 'inner', 'speaker', 'what',
                        'wakati',
                        'emotion']
        data = dict.fromkeys(data_columns)
        data['original'] = True
        data['koma'] = 0
        data['emotion'] = 'None'
        data['what'] = '[PAD]'
        # data['wakati'] = np.array([np.zeros((1, 300))]).reshape(300)
        data['wakati'] = ['[PAD]']
        story_main_max = self.data[touch_name].story_main_num.max()
        story_main_min = self.data[touch_name].story_main_num.min()
        sep = []
        cnt = 0
        now_id = 0
        print("seq_len : {}".format(seq_len))
        for i in range(story_main_min, story_main_max + 1):
            data['story_main_num'] = i
            data['story_sub_num'] = 0

            for seq in range(0, seq_len - 1):
                data['id'] = now_id
                now_id += 1
                sep.append(pd.DataFrame.from_dict([data]))

            cnt += 1

            df = self.data[touch_name][(self.data[touch_name].story_main_num == i) & (self.data[touch_name].story_sub_num == 0)]
            df.id = df.id + (seq_len - 1) * cnt
            sep.append(df)
            now_id = df.id.max() + 1

            data['story_sub_num'] = 1

            for seq in range(0, seq_len - 1):
                data['id'] = now_id
                now_id += 1
                sep.append(pd.DataFrame.from_dict([data]))

            cnt += 1

            df = self.data[touch_name][(self.data[touch_name].story_main_num == i) & (self.data[touch_name].story_sub_num == 1)]
            df.id = df.id + (seq_len - 1) * cnt
            sep.append(df)
            now_id = df.id.max() + 1

        new_data_set = pd.concat([s for s in sep], sort=False)
        self.data[touch_name] = new_data_set.reset_index(drop=True)


#amanga4koma = manga4koma(to_zero_pad=False)
#print(amanga4koma.data['gyagu'].what)
#print(amanga4koma.data['gyagu'].tokenized)
#print(amanga4koma.data['gyagu'].bert_tokenized)
#embedding = Embbedding(amanga4koma.tokenized, "../models/d2v_manga109.model", mode='d2v', word_base=True)
#print(embedding.embed['gyagu'].size)
#print(embedding.embed['gyagu'])
#print(amanga4koma.data['gyagu'].what)
#input_ids = torch.tensor(amanga4koma.bert_tokenized['gyagu']['input_ids'])
#print(bert_tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))

