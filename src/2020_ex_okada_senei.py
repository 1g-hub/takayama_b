# coding: utf-8
from utils.history import History
from my_network.pytorch_mlp import ClassificationNet, MLP3Net
import torch
from transformers import BertConfig, BertModel
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import optuna
from sklearn.metrics import classification_report
import numpy as np
from utils.visualizer import Visualizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import BertTokenizer
# ========
# GLOBAL 変数

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

bert_tokenizer = BertTokenizer('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt',
                                   do_lower_case=False, do_basic_tokenize=False,
                                   config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')


# ========
# 最終層のみのfine-tuning
class Net(nn.Module):
    config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')

    def __init__(self):
        super(Net, self).__init__()
        self.bert = BertModel.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin',
                                              config=self.config)
        self.classifier = MLP3Net(in_dim=768)

        # Bertの1〜11段目は更新せず、12段目とSequenceClassificationのLayerのみトレーニングする。
        # 一旦全部のパラメータのrequires_gradをFalseで更新
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        # Bert encoderの最終レイヤのrequires_gradをTrueで更新
        for name, param in self.bert.encoder.layer[-1].named_parameters():
            param.requires_grad = True

    def forward(self, x):
        bert_out = self.bert(x)[0] # 最後の隠れ層
        out = self.classifier(bert_out[:, 0, :]) # [CLS] に相当する部分のみ使う
        return out


def get_data_loader(batch_size):
    train_valid_data_set = pd.read_csv(
        '../dataset/train_develop_ja.csv',
        index_col=0
    )

    train_valid_data_set.wakati = [w.split(' ') for w in train_valid_data_set.wakati.tolist()]

    i = 0
    for index, row in train_valid_data_set.iterrows():
        if len(row.wakati) > 510:
            train_valid_data_set = train_valid_data_set.drop(i)

        i += 1

    s = train_valid_data_set.wakati.tolist()
    train_valid_data_set['bert_tokenized'] = [b for b in bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True,
                                                                                           add_special_tokens=True,
                                                                                           is_pretokenized=True)['input_ids']]


    for index, row in train_valid_data_set.iterrows():
        if len(row['bert_tokenized']) > 512:
            print("WWWWWWWWW")


    test_data_set = pd.read_csv(
        '../dataset/test_develop_ja.csv',
        index_col=0
    )
    test_data_set.wakati = [w.split(' ') for w in test_data_set.wakati.tolist()]

    i = 0
    for index, row in test_data_set.iterrows():
        if len(row.wakati) > 510:
            test_data_set = test_data_set.drop(i)

        i += 1


    s = test_data_set.wakati.tolist()
    test_data_set['bert_tokenized'] = [b for b in bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True,
                                                                                          add_special_tokens=True,
                                                                                          is_pretokenized=True)['input_ids']]
    del s
    # train, valid に分ける
    train_data_set, valid_data_set = train_test_split(
        train_valid_data_set,
        test_size=0.2,
        random_state=25,
        shuffle=True
    )
    y_train = np.identity(2)[[train_data_set.target]]
    y_valid = np.identity(2)[[valid_data_set.target]]
    y_test = np.identity(2)[[test_data_set.target]]

    # Tensor型へ (labelのデータ型はCrossEntrotyLoss:long ,others:float)
    X_train = torch.tensor(train_data_set['bert_tokenized'].values.tolist())
    y_train = torch.tensor(y_train, requires_grad=True).long()

    X_valid = torch.tensor(valid_data_set['bert_tokenized'].values.tolist())
    y_valid = torch.tensor(y_valid, requires_grad=True).long()

    X_test = torch.tensor(test_data_set['bert_tokenized'].values.tolist())
    y_test = torch.tensor(y_test, requires_grad=True).long()

    # 各DataLoaderの準備
    train = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)

    valid = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)

    test = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    del X_train, X_valid, X_test, y_train, y_valid, y_test

    return train_loader, valid_loader, test_loader, test_data_set

class Manga4koma_Experiment():
    def __init__(self, batch_size=16, epochs=50):
        self.epochs = epochs
        self.batch_size = batch_size

        self.train_loader, self.valid_loader, self.test_loader, self.test_data_set = get_data_loader(self.batch_size)

        self.data_loaders = {'train': self.train_loader, 'valid': self.valid_loader}

        self.criterion = torch.nn.CrossEntropyLoss()

        self.log_path = './result_okada_sensei_last_layer_2.txt'
        self.new_model_path = '../models/bert/My_Japanese_transformers/okada_sensei_last_layer_2.bin'
        self.touch_name = 'okada_sensei'
        self.reset_count()


        global log_f

    def manga4koma_train_sequencial(self):
        pass

    def manga4koma_train(self, lr, is_study=False):
        self.is_study = is_study

        self.history = History()
        self.net = Net().to(device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        if self.is_study == False:
            log_f = open(self.log_path, 'a', encoding='utf-8')
            print("best:lr {}".format(lr), file=log_f)
        for epoch in range(self.epochs):
            time_start = time.time()

            for phase in ['train', 'valid']:

                if phase == 'train':
                    self.net.train()
                else:
                    self.net.eval()

                self.reset_count()

                for x, y in self.data_loaders[phase]:

                    x = Variable(x).to(device)
                    y = Variable(y).to(device)

                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        y_pred = self.net(x)
                        _, predicted = torch.max(y_pred.data, 1)
                        self.total += y.size(0)
                        # loss 計算・加算
                        loss = self.criterion(y_pred, y.argmax(1))
                        self.total_loss += loss.item()
                        # 正解数 加算
                        self.correct += (predicted == y.argmax(1)).sum().item()
                        # 2 x 2 matrix 更新
                        for i in range(len(predicted)):
                            self.c_mat[torch.max(y.data, 1)[1][i]][predicted[i]] += 1

                        # 訓練時のみバックプロパゲーション
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                # ロスの合計を len(train_loader)で割る
                mean_loss = self.total_loss / len(self.data_loaders[phase])
                acc = (self.correct / self.total)
                f1 = self.cal_F1()


                # Historyに追加
                self.history.update(phase, mean_loss, acc, f1)

                if (phase == 'valid') & (self.history.enable_save) & (self.is_study is False):
                    self.save_model()

                # Validation 結果
                if phase == 'valid':
                    print("---Validation---")
                else:
                    print("---TRAIN---")
                print(self.c_mat)
                print("Acc : %.4f" % acc)
                print("F1: {0}".format(f1))



            time_finish = time.time() - time_start
            print("====================================")
            print("EPOCH : {0} / {1}".format(epoch + 1, self.epochs))
            print("残り時間 : {0}".format(time_finish * (self.epochs - epoch)))
            print("VAL_LOSS : {} \nVAL_ACCURACY : {}\n\n".format(self.history.history['valid']['loss'][-1],
                                                             self.history.history['valid']['acc'][-1]))

            if self.is_study == False:
                print("EPOCH : {0} / {1}".format(epoch + 1, self.epochs), file=log_f)
                print("VAL_LOSS : {} \nVAL_ACCURACY : {}\nVAL_F1 : {}\n".format(self.history.history['valid']['loss'][-1],
                                                                                self.history.history['valid']['acc'][-1],
                                                                                self.history.history['valid']['f1'][-1]), file=log_f)

        if self.is_study == False:
            v = Visualizer(self.touch_name, self.history.history)
            v.visualize()

            log_f.close()

        return self.history.best['valid']['acc']

    def manga4koma_test(self):
        log_f = open(self.log_path, 'a', encoding='utf-8')
        self.net = Net().to(device)
        self.load_model()
        self.reset_count()
        test_index = 0
        label = []
        pred = []
        self.net.eval()
        with torch.no_grad():
            for x_test, y_test in self.test_loader:
                x_test = Variable(x_test).to(device)
                y_test = Variable(y_test).to(device)

                self.optimizer.zero_grad()

                y_pred = self.net(x_test)

                _, predicted = torch.max(y_pred.data, 1)

                self.correct += (predicted == torch.max(y_test.data, 1)[1]).sum().item()

                self.total += y_test.size(0)

                pred.append(predicted[0])
                label.append(torch.max(y_test.data, 1)[1][0])
                # 2 x 2 matrix 更新
                for i in range(len(predicted)):
                    self.c_mat[torch.max(y_test.data, 1)[1][i]][predicted[i]] += 1

                test_index = min(len(self.test_data_set) - 1, test_index + 1)

            print("------------------------test acc------------------------", file=log_f)
            print("Test Acc : %.4f" % (self.correct / self.total), file=log_f)
            print("correct: {0}, total: {1}".format(self.correct, self.total), file=log_f)
            print("------------------------------------------------", file=log_f)
        d = classification_report([la.tolist() for la in label], [pr.tolist() for pr in pred],
                                  target_names=['0', '1'],
                                  output_dict=True)
        df = pd.DataFrame(d)
        print(df, file=log_f)
        print("正例のF1値 : " + str(self.cal_F1()), file=log_f)
        log_f.close()

    def save_model(self):
        torch.save(self.net.state_dict(), self.new_model_path)
        print('\nbest score updated, Pytorch model was saved!! acc:{}\n'.format(self.history.best['valid']['acc']))

    def load_model(self):
        load_weights = torch.load(self.new_model_path,
                                  map_location={'cuda:0': 'cpu'})
        self.net.load_state_dict(load_weights)

    def reset_count(self):
        self.total_loss = 0
        self.total = 0
        self.correct = 0
        self.c_mat = np.zeros((2, 2), dtype=int)

    def cal_F1(self):
        # 正例の F1値を返す
        c_precision = self.c_mat[0][0] / (1e-09 + self.c_mat[0][0] + self.c_mat[0][1])
        c_recall = self.c_mat[0][0] / (1e-09 + self.c_mat[0][0] + self.c_mat[1][0])
        c_f1 = (2 * c_recall * c_precision) / (1e-09 + c_recall + c_precision)
        return c_f1

    def objective_variable(self):

        def objective(trial):
            lr = trial.suggest_loguniform('lr', 1e-6, 1e-5)
            print("suggest lr = {}".format(lr))
            best_valid_acc = self.manga4koma_train(lr=lr, is_study=True)
            error = 1 - best_valid_acc
            return error

        return objective

    def optuna_optimize(self):
        study = optuna.create_study()
        study.optimize(self.objective_variable(), n_trials=2)
        return study.best_trial

def main():
    print("experience start device:{}".format(device))

    ex = Manga4koma_Experiment(batch_size=16, epochs=100)

    trial = ex.optuna_optimize()

    print('Best trial:')
    print('  Value: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    ex.manga4koma_train(lr=trial.params['lr'], is_study=False)
    ex.manga4koma_test()

if __name__ == '__main__':
    main()