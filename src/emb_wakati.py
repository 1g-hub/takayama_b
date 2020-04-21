# coding: utf-8
from gensim.models import Doc2Vec

import numpy as np
import pandas as pd
from utils import nlp_util as nlp

save_path = "model/d2v_manga109.model"

d2v_model = Doc2Vec.load(save_path)
TOUCH_NAME = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
# 拡張後のデータ読み込み
for touch_name in TOUCH_NAME:
    data_set = pd.read_csv('data/old/' + touch_name + '_augmentation.csv', index_col=0, dtype={'original': bool})
    data_set.wakati = nlp.jumanpp_wakati2array(data_set.wakati)
    print("{} : ".format(touch_name))
    print("== Convert Start ==")
    data_set.wakati = [d2v_model.infer_vector(w) for w in data_set.wakati]

    pd.to_pickle(data_set.wakati,"data/" + touch_name + "_emb_manga109.pkl")
    print("=== SAVE FINISH===")