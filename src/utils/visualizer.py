# coding: utf-8
import seaborn as sb
import os
import matplotlib.pyplot as plt
import time
import datetime

class Visualizer():
    def __init__(self, touch_name, history):
        self.today, self.path = self.get_today()
        self.touch_name = touch_name
        self.t_loss = history['train']['loss']
        self.t_acc = history['train']['acc']
        self.t_f1 = history['train']['f1']
        self.v_loss = history['valid']['loss']
        self.v_acc = history['valid']['acc']
        self.v_f1 = history['valid']['f1']
        self.make_dir()
    def get_today(self):
        today = datetime.datetime.now()  # 今日の日付を取得
        # 日付をフォルダ名に整形
        Y = str(today.year)
        M = str(today.month)
        MM = M.zfill(2)
        D = str(today.day)
        DD = D.zfill(2)
        H = str(today.hour)
        HH = H.zfill(2)
        MI = str(today.minute).zfill(2)
        fname = Y + '_' + MM + '_' + DD + '_' + HH + '_' + MI  # フォルダ名を定義
        path = '../results/' + fname  # 生成フォルダのパス

        return fname, path

    def make_dir(self):
        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.path + '/' + self.touch_name, exist_ok=True)

    def visualize(self):
        # === 図の作成, 保存 ===
        plt.plot(self.t_loss, label="train_loss")
        plt.plot(self.v_loss, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(self.touch_name + "_train_val_loss")
        plt.legend()
        plt.savefig(self.path + '/' + self.touch_name + '/' + self.touch_name + "_train_val_loss" + ".png")
        plt.clf()

        plt.plot(self.t_acc, label="train_accuracy")
        plt.plot(self.v_acc, label="val_accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(self.touch_name + "_train_val_acc")
        plt.legend()
        plt.savefig(self.path + '/' + self.touch_name + '/' + self.touch_name + "_train_val_acc" + ".png")
        plt.clf()

        plt.plot(self.t_f1, label="train_f1")
        plt.plot(self.v_f1, label="val_f1")
        plt.xlabel("epoch")
        plt.ylabel("f1")
        plt.title(self.touch_name + "_train_val_f1")
        plt.legend()
        plt.savefig(self.path + '/' + self.touch_name + '/' + self.touch_name + "_train_val_f1" + ".png")
        plt.clf()


