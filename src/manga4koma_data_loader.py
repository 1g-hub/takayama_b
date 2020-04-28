# coding: utf-8

from sklearn.model_selection import train_test_split

import numpy as np

import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

seq_len = [2, 3, 4, 5, 6]

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

def create_seq(data, batch_size=16, shuffle=True):
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
