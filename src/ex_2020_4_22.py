# coding: utf-8
import time
import statistics
import math

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

import seaborn as sb
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from itertools import chain

from my_network import pytorch_self_attention as net
from utils import nlp_util as nlp

from functools import partial

import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC
from sklearn import datasets

def objective(X, y, trial):
    """最小化する目的関数"""
    params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'sigmoid']),
        'C': trial.suggest_loguniform('C', 1e+0, 1e+2),
        'gamma': trial.suggest_loguniform('gamma', 1e-2, 1e+1),
    }

    # モデルを作る
    model = SVC(**params)

    # 自身の問題に合わせて
    # クラス化した方が良い？？

    # 5-Fold CV / Accuracy でモデルを評価する
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, X=X, y=y, cv=kf)
    # 最小化なので 1.0 からスコアを引く
    return 1.0 - scores['test_score'].mean()

def main():
    # 乳がんデータセットを読み込む
    dataset = datasets.load_breast_cancer()
    X, y = dataset.data, dataset.target
    # 目的関数にデータを適用する
    f = partial(objective, X, y)
    # 最適化のセッションを作る
    study = optuna.create_study()
    # 100 回試行する
    study.optimize(f, n_trials=100)
    # 最適化したパラメータを出力する
    print('params:', study.best_params)


if __name__ == '__main__':
    main()