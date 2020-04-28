# coding: utf-8
from gensim.models import Doc2Vec
import torch
import numpy as np
from collections import defaultdict
import pandas as pd
from utils import nlp_util as nlp

TOUCH_NAME = ["gyagu", "shoujo", "shounen", "seinen", "moe"]

class Embbedding():
    def __init__(self, data, model_path, mode='d2v', word_base=False):
        self.mode = mode
        self.data = data # tokenize されたものを渡す
        self.model_path = model_path
        self.word_base = word_base # 単語ごとに
        self.embed = defaultdict(torch.Tensor)
        if mode == 'bert':
            self.emb_bert()
        elif mode == 'd2v':
            self.emb_d2v()

    def emb_d2v(self):
        model = Doc2Vec.load(self.model_path)

        # 拡張後のデータ読み込み
        for touch_name in TOUCH_NAME:
            if self.word_base:
                li = []
                for s in self.data[touch_name]:
                    tmp = torch.tensor([model.infer_vector(s) if w != "[PAD]" else np.array([np.zeros((1,300))]).reshape(300) for w in s])
                    li.append(tmp)
                # error
                self.embed[touch_name] = torch.stack([t for t in li], dim=0)
            else:
                self.embed[touch_name] = torch.tensor([model.infer_vector(s) for s in self.data])


    def emb_bert(self):
        pass
