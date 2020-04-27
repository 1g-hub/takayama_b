# coding: utf-8
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import nlp_util as nlp

import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig


bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt',
                                   do_lower_case=False, do_basic_tokenize=False,
                                   config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')

class manga4koma():
    def __init__(self):
        self.TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
        self.CONV_TOUCH = {"gyagu": "ギャグタッチ", "shoujo": "少女漫画タッチ", "shounen": "少年漫画タッチ", "seinen": "青年漫画タッチ", "moe": "萌え系タッチ"}
        self.EMB_MODE = ["d2v", "bert"]
        self.EMB_DIM = {"d2v": 300, "bert": 768}
        self.CONV_EMO = {"ニュートラル": 'neutral', "驚愕": 'kyougaku', "喜楽": 'kiraku', "恐怖": 'kyouhu', "悲哀": 'hiai', "憤怒": 'hunnu', "嫌悪": 'keno'}
        self.SEQ_LEN = [2, 3, 4, 5, 6]
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

    def __tokenize(self):
        for touch_name in self.TOUCH_NAME_ENG:
            s = self.data[touch_name].wakati.tolist()
            self.tokenized[touch_name] = s
            self.bert_tokenized[touch_name] = bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)


def main():
    amanga4koma = manga4koma()
    print(amanga4koma.data['gyagu'].wakati)
    input_ids = torch.tensor(amanga4koma.bert_tokenized['gyagu']['input_ids'])
    print(bert_tokenizer.convert_ids_to_tokens(input_ids[0].tolist()))


if __name__ == '__main__':
    main()
