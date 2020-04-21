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
from .utils import nlp_util as nlp

from sklearn.metrics import classification_report, accuracy_score
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from itertools import chain

from .my_network import pytorch_self_attention as net


BATCH_SIZE = 16 # バッチサイズ
EMB_DIM = 300 # 単語の埋め込み次元数
HIDDEN_DIM = 128 # LSTMの隠れ層の次元数
TAG_SIZE = 2 # 今回はネガポジ判定を行うのでネットワークの最後のサイズは2
DA = 64 # AttentionをNeural Networkで計算する際の重み行列のサイズ
R = 3 # Attentionを３層重ねて見る


def zero_padding(data_set, seq_len=3):
    data_columns = ['id', 'original', 'story_main_num', 'story_sub_num', 'koma', 'who', 'inner', 'speaker', 'what',
                    'wakati',
                    'emotion']
    data = dict.fromkeys(data_columns)
    data['original'] = True
    data['koma'] = 0
    data['what'] = '<pad>'
    data['wakati'] = np.array([np.zeros((1,300))]).reshape(300)
    story_main_max = data_set.story_main_num.max()
    story_main_min = data_set.story_main_num.min()
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

        df = data_set[(data_set.story_main_num == i) & (data_set.story_sub_num == 0)]
        df.id = df.id + (seq_len - 1) * cnt
        sep.append(df)
        now_id = df.id.max() + 1

        data['story_sub_num'] = 1

        for seq in range(0, seq_len - 1):
            data['id'] = now_id
            now_id += 1
            sep.append(pd.DataFrame.from_dict([data]))

        cnt += 1

        df = data_set[(data_set.story_main_num == i) & (data_set.story_sub_num == 1)]
        df.id = df.id + (seq_len - 1) * cnt
        sep.append(df)
        now_id = df.id.max() + 1

    new_data_set = pd.concat([s for s in sep], sort=False)
    new_data_set = new_data_set.reset_index(drop=True)

    return new_data_set

def create_train_valid(data_set, max_story_main_num=4, seq_len=3, val_size=0.2, random_seed=25, shuffle=True, augmentation=False, aug_rate=[2, 0.5]):
    train_data_set = data_set[(data_set.story_main_num <= max_story_main_num)]
    f_trains = train_data_set[(train_data_set.original)]

    x = []
    y = []

    for f_index, f_train_data in f_trains.iterrows():

        front_in = []

        give_up = False

        for seq in range(seq_len - 1):
            next = f_trains[(f_trains.id - seq == f_train_data.id) & (f_trains.story_sub_num == f_train_data.story_sub_num)]
            if len(next) == 0:
                give_up = True
                break
            else:
                next = next.iloc[0]
            front_in.append(next)

        if give_up:
            continue

        third_ins = train_data_set[
            (train_data_set.id - 1 == front_in[-1].id) & (train_data_set.story_sub_num == front_in[-1].story_sub_num)]


        if len(third_ins) == 0:
            continue
        else:
            if third_ins.iloc[0].emotion != '喜楽':
                pass
                # third_ins = third_ins.sample(frac=min(1, math.sqrt(others/kiraku)))
                # if len(third_ins) == 0:
                #     continue
            # else:
            #     if kiraku > others:
            #         over_samples = third_ins.sample(frac=min(1, math.sqrt(kiraku / others) - 1))
            #         third_ins = pd.concat([third_ins, over_samples])
            # itr += len(third_ins)


        for t_index, third_in in third_ins.iterrows():
            # print("===入力文===")
            # print(first_in.what + " " + second_in.what + " " + third_in.what)
            fronts = np.stack([front.wakati for front in front_in], axis=0)

            if seq_len == 2:
                input = np.stack([front_in[0].wakati, third_in.wakati], axis=0)
            elif seq_len == 3:
                input = np.stack([front_in[0].wakati, front_in[1].wakati, third_in.wakati], axis=0)
            elif seq_len == 4:
                input = np.stack([front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, third_in.wakati], axis=0)
            elif seq_len == 5:
                input = np.stack([front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, front_in[3].wakati, third_in.wakati], axis=0)
            elif seq_len == 6:
                input = np.stack([front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, front_in[3].wakati, front_in[4].wakati,third_in.wakati], axis=0)
            #input = np.stack([fronts, third_in.wakati], axis=0)
            x.append(input)
            y_np = np.identity(2)[0 if third_in.emotion == '喜楽' else 1]
            y.append(y_np)

    x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=val_size, random_state=random_seed, shuffle=shuffle)

    return x_t, x_v, y_t, y_v

def create_test_data_loader(test_data_set):
    x = []
    y = []

    for f_index, f_test_data in test_data_set.iterrows():

        front_in = []

        give_up = False

        for seq in range(seq_len - 1):
            next = test_data_set[
                (test_data_set.id - seq == f_test_data.id) & (test_data_set.story_sub_num == f_test_data.story_sub_num)]
            if len(next) == 0:
                give_up = True
                break
            else:
                next = next.iloc[0]
            front_in.append(next)

        if give_up:
            continue

        third_ins = test_data_set[
            (test_data_set.id - 1 == front_in[-1].id) & (test_data_set.story_sub_num == front_in[-1].story_sub_num)]

        if len(third_ins) == 0:
            continue

        for t_index, third_in in third_ins.iterrows():
            # print("===入力文===")
            # print(first_in.what + " " + second_in.what + " " + third_in.what)
            fronts = np.stack([front.wakati for front in front_in], axis=0)
            if seq_len == 2:
                input = np.stack([front_in[0].wakati, third_in.wakati], axis=0)
            elif seq_len == 3:
                input = np.stack([front_in[0].wakati, front_in[1].wakati, third_in.wakati], axis=0)
            elif seq_len == 4:
                input = np.stack([front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, third_in.wakati], axis=0)
            elif seq_len == 5:
                input = np.stack(
                    [front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, front_in[3].wakati, third_in.wakati],
                    axis=0)
            elif seq_len == 6:
                input = np.stack(
                    [front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, front_in[3].wakati, front_in[4].wakati,
                     third_in.wakati], axis=0)
            # input = np.stack([fronts, third_in.wakati], axis=0)
            x.append(input)
            y_np = np.identity(2)[0 if third_in.emotion == '喜楽' else 1]
            y.append(y_np)

    X_test = torch.tensor(x, requires_grad=True).float()
    y_test = torch.tensor(y, requires_grad=True).long()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=1, shuffle=False)

    return test_loader

def create_data_loader(x_t, x_v, y_t, y_v, batch_size=16, shuffle=True):
    # Tensor型へ (labelのデータ型はCrossEntrotyLoss:long ,others:float)
    X_train = torch.tensor(x_t, requires_grad=True).float()
    y_train = torch.tensor(y_t, requires_grad=True).long()
    X_valid = torch.tensor(x_v, requires_grad=True).float()
    y_valid = torch.tensor(y_v, requires_grad=True).long()
    # 各DataLoaderの準備
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(TensorDataset(X_valid, y_valid), batch_size=batch_size, shuffle=shuffle)

    return train_loader, valid_loader

def create_miria(data, batch_size=16, shuffle=True):
    f_trains = data[(data.original)]

    x = []
    y = []

    for f_index, f_train_data in f_trains.iterrows():

        front_in = []

        give_up = False

        for seq in range(seq_len - 1):
            next = f_trains[
                (f_trains.id - seq == f_train_data.id) & (f_trains.story_sub_num == f_train_data.story_sub_num)]
            if len(next) == 0:
                give_up = True
                break
            else:
                next = next.iloc[0]
            front_in.append(next)

        if give_up:
            continue

        third_ins = data[
            (data.id - 1 == front_in[-1].id) & (data.story_sub_num == front_in[-1].story_sub_num)]

        if len(third_ins) == 0:
            continue


        for t_index, third_in in third_ins.iterrows():
            # print("===入力文===")
            # print(first_in.what + " " + second_in.what + " " + third_in.what)
            fronts = np.stack([front.wakati for front in front_in], axis=0)

            if seq_len == 2:
                input = np.stack([front_in[0].wakati, third_in.wakati], axis=0)
            elif seq_len == 3:
                input = np.stack([front_in[0].wakati, front_in[1].wakati, third_in.wakati], axis=0)
            elif seq_len == 4:
                input = np.stack([front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, third_in.wakati], axis=0)
            elif seq_len == 5:
                input = np.stack(
                    [front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, front_in[3].wakati, third_in.wakati],
                    axis=0)
            elif seq_len == 6:
                input = np.stack(
                    [front_in[0].wakati, front_in[1].wakati, front_in[2].wakati, front_in[3].wakati, front_in[4].wakati,
                     third_in.wakati], axis=0)
            # input = np.stack([fronts, third_in.wakati], axis=0)
            x.append(input)
            y_np = np.identity(2)[0 if third_in.emotion == '喜楽' else 1]
            y.append(y_np)

    # Tensor型へ (labelのデータ型はCrossEntrotyLoss:long ,others:float)
    X_train = torch.tensor(x, requires_grad=True).float()
    y_train = torch.tensor(y, requires_grad=True).long()
    # 各DataLoaderの準備
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=shuffle)

    return train_loader


def Cal_F1(c_mat, mode='macro'):
    c_precision = c_mat[0][0] / (1e-09 + c_mat[0][0] + c_mat[0][1])
    c_recall = c_mat[0][0] / (1e-09 + c_mat[0][0] + c_mat[1][0])
    c_f1 = (2 * c_recall * c_precision) / (1e-09 + c_recall + c_precision)
    if mode == 'positive':
        return c_f1
    nc_precision = c_mat[1][1] / (1e-09 + c_mat[1][1] + c_mat[1][0])
    nc_recall = c_mat[1][1] / (1e-09 + c_mat[1][1] + c_mat[0][1])
    nc_f1 = (2 * nc_recall * nc_precision) / (1e-09 + nc_recall + nc_precision)
    if mode == 'negative':
        return nc_f1

    if mode == 'macro':
        return (c_f1 + nc_f1) / 2


TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
CONV_TOUCH = {"gyagu":"ギャグタッチ", "shoujo":"少女漫画タッチ", "shounen":"少年漫画タッチ", "seinen":"青年漫画タッチ", "moe":"萌え系タッチ"}
EMB_MODE = ["d2v"]
EMB_DIM = {"d2v":300,"bert":768}
CONV_EMO = {"ニュートラル": 'neutral', "驚愕": 'kyougaku', "喜楽": 'kiraku', "恐怖": 'kyouhu', "悲哀": 'hiai', "憤怒": 'hunnu', "嫌悪": 'keno'}

seq_lens = [2, 3, 4, 5, 6]

# 各タッチについて実験
for touch_name in TOUCH_NAME_ENG:
    # lrの設定を2乗から1乗に
    # pad を追加
    # epoch数を50に?
    # オーバーアンダーサンプリング比率変化

    print("TOUCH : {} \n!!!===START===!!!".format(touch_name))



    for seq_len in seq_lens:

        log_file_path = "result/" + touch_name + "/" + str(seq_len) + "_" + touch_name + "_log.txt"

        log_f = open(log_file_path, 'w', encoding='utf-8')

        """
        ==========================================================
        [1] データの読み込み・分かち書きの分散表現化
        ==========================================================
        """
        data_set = pd.read_csv(
            'data/' + touch_name + '_augmentation.csv',
            index_col=0,
            dtype={'original': bool},
            usecols=lambda x: x is not 'index'
        )
        # 分かち書き分散表現化
        data_set.wakati = pd.read_pickle('data/' + touch_name + '_emb_manga109.pkl')
        original_test = data_set[(data_set.original) & (data_set.story_main_num >= 5)]

        # <pad>付け足し
        data_set = zero_padding(data_set, seq_len=seq_len)
        print(len(data_set[data_set.what == '<pad>']))
        print(data_set.what)

        # テストデータを分ける
        test_data_set = data_set[(data_set.original) & (data_set.story_main_num >= 5)]
        originals = data_set[(data_set.original)]
        # cos_boarder = 0.4
        # for index, original in originals.iterrows():
        #     augs = data_set[(data_set.original == False) & (data_set.story_main_num < 5) & (data_set.id == original.id)]
        #     for index_aug, aug in augs.iterrows():
        #         if (nlp.cos_similarity(aug.wakati, original.wakati)) < cos_boarder:
        #             print(nlp.cos_similarity(aug.wakati, original.wakati),aug.what)
        #             data_set = data_set.drop(data_set[(data_set.what == aug.what) & (data_set.id == aug.id)].index)



        print("====================", file=log_f)
        print("touch:{}\n".format(CONV_TOUCH[touch_name]), file=log_f)
        print("seq_len:{}".format(seq_len), file=log_f)

        """
        ==========================================================
        [2] Pytorch入力用にデータの整形
        ==========================================================
        """
        test_acc_history = []
        test_f1_history = []
        test_mcc_history = []

        train_cv_acc_history = []
        train_cv_f1_history = []
        train_cv_mcc_history = []

        val_cv_acc_history = []
        val_cv_f1_history = []
        val_cv_mcc_history = []

        # 今のレポート
        train_data_set = data_set[(data_set.story_main_num < 5)]
        f_trains = train_data_set[(train_data_set.original)]

        x_t, x_v, y_t, y_v = create_train_valid(data_set, val_size=0.2, random_seed=42, seq_len=seq_len)

        train_loader, valid_loader = create_data_loader(x_t, x_v, y_t, y_v, batch_size=16, shuffle=True)

        # train_data_set = data_set[(data_set.story_main_num < 3)]
        # valid_data_set = data_set[(data_set.story_main_num == 4) | (data_set.story_main_num == 3)]
        # train_loader = create_miria(train_data_set)
        # valid_loader = create_miria(valid_data_set)
        """
        ==========================================================
        [3] ネットワークに入力させて学習する
        ==========================================================
        """



        # === ネットのインスタンス化 ===
        encoder = net.BiLSTMEncoder(300, 128)
        classifier = net.SelfAttentionClassifier(128, 64, 3, 2)

        # 複数のモデルを from itertools import chain で囲えばoptimizerをまとめて1つにできる
        optimizer = torch.optim.Adam(chain(encoder.parameters(), classifier.parameters()), lr=0.00001)


        # === クラスの重み(必要であれば) === & (data_set.original)
        k_num = len(train_data_set[(train_data_set.emotion == '喜楽') & (train_data_set.story_main_num < 5) & (train_data_set.original)])
        other_num = len(train_data_set[(train_data_set.emotion != '喜楽') & (train_data_set.story_main_num < 5) & (train_data_set.original)])

        kiraku = 1 / (1e-09 + k_num)
        others = 1 / (1e-09 + other_num)
        w = torch.tensor([kiraku, others]).float()
        # w = torch.tensor([100.0, 1.0]).float()

        print("w(喜楽, その他) : {}".format(w))

        print("w(喜楽, その他) : {}".format(w), file=log_f)


        # === lossと最適化手法の設定 ===
        criterion = torch.nn.CrossEntropyLoss()

        epoch_num = 100

        print("epoch数:{}".format(epoch_num), file=log_f)
        print("学習率:{}".format(0.00001), file=log_f)

        train_loss_history = []
        train_acc_history = []
        train_f1_history = []
        train_mcc_history = []
        val_loss_history = []
        val_acc_history = []
        val_f1_history = []
        val_mcc_history = []
        val_recall_history = []
        val_best_acc = 0
        val_best_f1 = 0
        val_best_mcc = -1

        train_best_acc = 0
        train_best_f1 = 0
        train_best_mcc = -1

        test_id_min = test_data_set.id.min()
        test_id_max = test_data_set.id.max() + 1
        # print("train_id_max : {}".format(train_id_max - 1))
        # print("valid_id_max : {}".format(valid_id_max - 1))
        print("test_id_max : {}".format(test_id_max - 1))

        lr_history = []

        for epoch in range(epoch_num + 1):
            # === Train ===

            time_start = time.time()


            total_loss = 0
            total = 0
            correct = 0
            itr = 0

            c_mat = np.zeros((2, 2), dtype=int)


            for x_train, y_train in train_loader:
                x_train = x_train.view(len(x_train), seq_len, -1)  # input.shape = [batch_size, seq_len, 300]

                x_train = Variable(x_train)

                y_train = y_train.view(len(y_train),2)
                y_train = Variable(y_train)
                encoder.zero_grad()
                classifier.zero_grad()
                optimizer.zero_grad()

                out = encoder(x_train)
                score, attn = classifier(out)

                _, predicted = torch.max(score, 1)
                # print("predicted : {}".format(predicted))
                # print("ans : {}".format(torch.max(y_train.data, 1)[1]))
                correct += (predicted == torch.max(y_train.data, 1)[1]).sum().item()
                loss = criterion(score, y_train.argmax(1))
                total += y_train.size(0)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()


                for i in range(len(predicted)):
                    c_mat[torch.max(y_train.data, 1)[1][i]][predicted[i]] += 1

            # ロスの合計を len(train_loader)で割る
            train_mean_loss = total_loss / len(train_loader)
            train_acc = (correct / total)
            # Historyに追加
            train_loss_history.append(train_mean_loss) #epochごとのhistory
            train_acc_history.append(train_acc) #epochごとのhistory
            print("train loader itr {}".format(len(train_loader)))


            c_f1 = Cal_F1(c_mat, mode='positive')

            train_f1_history.append(c_f1)

            # === Validation ===
            total_loss = 0
            correct = 0
            itr = 0
            total = 0

            c_mat = np.zeros((2,2), dtype=int)

            with torch.no_grad():

                for x_valid, y_valid in valid_loader:
                    x_valid = x_valid.view(len(x_valid), seq_len, -1)  # input.shape = [1, 3, 300]

                    y_valid = y_valid.view(len(y_valid), 2)

                    encoder.zero_grad()
                    classifier.zero_grad()
                    optimizer.zero_grad()

                    out = encoder(x_valid)
                    score, attn = classifier(out)

                    _, predicted = torch.max(score, 1)

                    correct += (predicted == torch.max(y_valid.data, 1)[1]).sum().item()
                    loss = criterion(score, y_valid.argmax(1))

                    total_loss += loss.item()
                    total += y_valid.size(0)

                    for i in range(len(predicted)):
                        c_mat[torch.max(y_valid.data, 1)[1][i]][predicted[i]] += 1

                print("valid itr {}".format(itr))
                print(c_mat)

                c_f1 = Cal_F1(c_mat, mode='positive')
                val_acc = (correct / total)
                val_mean_loss = total_loss / len(valid_loader)

                # Historyに追加 1epochごとの
                val_loss_history.append(val_mean_loss)
                val_acc_history.append(val_acc)
                val_f1_history.append(c_f1)  #マクロ平均F1
                val_recall_history.append(c_mat[0][0]/(c_mat[0][0] + c_mat[0][1]))
                print("---Validation--- seq_len = {}".format(seq_len))
                print("Val Acc : %.4f" % val_acc)
                print("correct: {0}, total: {1} //// f1: {2}".format(correct, itr, c_f1))

            if c_f1 > val_best_f1:
                val_best_f1 = c_f1
                nlp.save_torch_model(net, touch_name + '_4_15_selfA_'+ str(seq_len))
                print('\nbest score updated, Pytorch model was saved!! f1:{}\n'.format(val_best_f1))
                train_best_acc = train_acc
                train_best_f1 = train_f1_history[-1]

                val_best_acc = val_acc

            time_finish = time.time() - time_start


            print("====================================")
            print("EPOCH : {0} / {1}".format(epoch + 1, epoch_num))
            print("VAL_LOSS : {} \nVAL_ACCURACY : {}".format(val_mean_loss, val_acc))
            time_nokori = (epoch_num - epoch) * time_finish
            print("{}における残り時間 : {} [min]".format(touch_name, time_nokori / 60))
            print("====================================", file=log_f)
            print("EPOCH : {}".format(epoch + 1), file=log_f)
            print("VAL_LOSS : {} \nVAL_ACCURACY : {}".format(val_mean_loss, val_acc), file=log_f)


        # === Best Modelを読み込んで予測===
        net = nlp.load_torch_model(touch_name + '_4_15_selfA_' + str(seq_len))


        # === Test ===
        test_loader = create_test_data_loader(test_data_set)
        correct = 0
        itr = 0
        total = 0
        label = []
        pred = []
        heat_matrix_7 = np.zeros((7,2), dtype=int)
        heat_matrix_2 = np.zeros((2, 2), dtype=int)
        test_index = 0
        print("###間違ったリスト###", file=log_f)
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.view(len(x_test), seq_len, -1)  # input.shape = [1, 3, 300]

                y_test = y_test.view(len(y_test), 2)

                out = encoder(x_test)
                score, attn = classifier(out)

                _, predicted = torch.max(score, 1)


                correct += (predicted == torch.max(y_test.data, 1)[1]).sum().item()

                total += y_test.size(0)



                if (predicted[0] != torch.max(y_test.data, 1)[1][0]):
                    print("# {}話-{}".format(original_test.iloc[test_index].story_main_num, original_test.iloc[test_index].story_sub_num),
                          file = log_f)
                    print("# 文: {0}\n正解 : {1} , 予測 : {2} / 元クラス : {3}".format(original_test.iloc[test_index].what,
                                                                              torch.max(y_test.data, 1)[1][0],
                                                                              predicted[0],
                                                                              original_test.iloc[test_index].emotion),
                          file=log_f)

                pred.append(predicted[0])
                label.append(torch.max(y_test.data, 1)[1][0])

                test_index = min(len(original_test)-1, test_index + 1)

            print("------------------------test acc------------------------", file=log_f)
            print("Test Acc : %.4f" % (correct / total), file=log_f)
            print("correct: {0}, total: {1}".format(correct, total), file=log_f)
            print("------------------------------------------------", file=log_f)


        d = classification_report(label, pred,
                                  target_names=['喜楽','その他'],
                                  output_dict=True)
        df = pd.DataFrame(d)
        print(df, file=log_f)

        # === 図の作成, 保存 ===
        plt.plot(train_loss_history, label="train_loss")
        plt.plot(val_loss_history, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(touch_name + "_train_val_loss")
        plt.legend()
        plt.savefig("result/" + touch_name + "/" + str(seq_len) + "_" + touch_name + "_train_val_loss" + ".png")
        #plt.show()

        plt.clf()

        plt.plot(train_acc_history, label="train_accuracy")
        plt.plot(val_acc_history, label="val_accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(touch_name + "_train_val_acc")
        plt.legend()
        plt.savefig("result/" + touch_name + "/" + str(seq_len) + "_" + touch_name + "_train_val_acc" + ".png")

        plt.clf()

        plt.plot(val_f1_history, label="val_f1")
        plt.xlabel("epoch")
        plt.ylabel("f1")
        plt.title(touch_name + "_evaluation")
        plt.legend()
        plt.savefig("result/" + touch_name + "/" + str(seq_len) + "_" + touch_name + "_evaluation" + ".png")
        #plt.show()

        plt.clf()




        # Train,

        # 公差検証の結果を出力
        print("\n=====公差検証結果=====\n", file=log_f)
        # Train
        train_cv_acc_mean = statistics.mean(train_acc_history)
        train_cv_f1_mean = statistics.mean(train_f1_history)

        train_cv_acc_max = max(train_acc_history)
        train_cv_f1_max = max(train_f1_history)
        print("Train Mean:", file=log_f)
        print("acc:{}\nf1:{}".format(train_cv_acc_mean,train_cv_f1_mean), file=log_f)
        print("Train Max:", file=log_f)
        print("acc:{}\nf1:{}".format(train_cv_acc_max, train_cv_f1_max), file=log_f)
        # Valid
        val_cv_acc_mean = statistics.mean(val_acc_history)
        val_cv_f1_mean = statistics.mean(val_f1_history)

        val_cv_acc_max = max(val_acc_history)
        val_cv_f1_max = max(val_f1_history)
        print("Valid Mean:", file=log_f)
        print("acc:{}\nf1:{}".format(val_cv_acc_mean, val_cv_f1_mean), file=log_f)
        print("Valid Max:", file=log_f)
        print("acc:{}\nf1:{}".format(val_cv_acc_max, val_cv_f1_max), file=log_f)

        plt.close()
        log_f.close()