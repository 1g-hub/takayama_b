# coding: utf-8
from collections import defaultdict


class History():
    def __init__(self):
        self.history = defaultdict(lambda: defaultdict(list))
        self.history['train']['loss'] = []
        self.history['train']['acc'] = []
        self.history['train']['f1'] = []
        self.history['valid']['loss'] = []
        self.history['valid']['acc'] = []
        self.history['valid']['f1'] = []
        self.best = defaultdict(lambda: defaultdict(float))
        self.best['train']['acc'] = 0.0 - 1e-5
        self.best['train']['f1'] = 0.0 - 1e-5
        self.best['valid']['acc'] = 0.0 - 1e-5
        self.best['valid']['f1'] = 0.0 - 1e-5

    def update(self, phase='train', new_loss=0.0, new_acc=0.0, new_f1=0.0):
        self.history[phase]['acc'].append(new_acc)
        self.history[phase]['f1'].append(new_f1)
        self.history[phase]['loss'].append(new_loss)

        if self.best[phase]['acc'] < new_acc:
            self.best[phase]['acc'] = new_acc

        if self.best[phase]['f1'] < new_f1:
            self.best[phase]['f1'] = new_f1

    def check(self, phase='train', new_f1=0.0):
        # モデル保存すべきかを返す
        return self.best[phase]['f1'] <= new_f1