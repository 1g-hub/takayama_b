# coding: utf-8
from manga4koma import manga4koma
from my_network.pytorch_mlp import ClassificationNet
import torch
from transformers import BertConfig, BertModel
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import optuna
from itertools import chain
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from utils import nlp_util as nlp
from utils.visualizer import Visualizer
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
EPOCHS = 200 # エポック数
BATCH_SIZE = 32 # バッチサイズ

manga_data = manga4koma()

config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')

bert_model = BertModel.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin', config=config)
classifier = ClassificationNet()

criterion = torch.nn.CrossEntropyLoss()

w = None

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def get_data_loader(touch_name):
    test_data_set = manga_data.data[touch_name][
        (manga_data.data[touch_name].story_main_num >= 5) & (manga_data.data[touch_name].original)]
    train_valid_data_set = manga_data.data[touch_name][(manga_data.data[touch_name].story_main_num < 5)]
    # train, valid に分ける
    train_data_set, valid_data_set = train_test_split(
        train_valid_data_set,
        test_size=0.2,
        random_state=25,
        shuffle=True
    )
    y_train = np.identity(2)[[0 if emo == '喜楽' else 1 for emo in train_data_set.emotion]]
    y_valid = np.identity(2)[[0 if emo == '喜楽' else 1 for emo in valid_data_set.emotion]]
    y_test = np.identity(2)[[0 if emo == '喜楽' else 1 for emo in test_data_set.emotion]]

    # Tensor型へ (labelのデータ型はCrossEntrotyLoss:long ,others:float)
    X_train = torch.tensor(train_data_set.bert_tokenized.values.tolist())
    y_train = torch.tensor(y_train, requires_grad=True).long()

    X_valid = torch.tensor(valid_data_set.bert_tokenized.values.tolist())
    y_valid = torch.tensor(y_valid, requires_grad=True).long()

    X_test = torch.tensor(test_data_set.bert_tokenized.values.tolist())
    y_test = torch.tensor(y_test, requires_grad=True).long()

    # 各DataLoaderの準備
    train = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)

    valid = TensorDataset(X_valid, y_valid)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, shuffle=True)

    test = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test, batch_size=1, shuffle=False)

    # === クラスの重み(必要であれば) ===
    k_num = len(train_data_set[(train_data_set.emotion == '喜楽')])
    other_num = len(train_data_set[(train_data_set.emotion != '喜楽')])
    kiraku = other_num / (1e-09 + k_num + other_num)
    others = k_num / (1e-09 + other_num + other_num)
    w = torch.tensor([kiraku, others]).float()

    criterion = torch.nn.CrossEntropyLoss(weight=w)

    return train_loader, valid_loader, test_loader, test_data_set

class Manga4koma_Trainer():
    def __init__(self, optimizer, criterion, train_loader, valid_loader, touch_name, is_study=False, batch_size=8):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = batch_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.touch_name = touch_name

        self.is_study = is_study

        self.train_loss_history = []
        self.train_acc_history = []
        self.train_f1_history = []
        self.valid_loss_history = []
        self.valid_acc_history = []
        self.valid_f1_history = []
        self.valid_best_acc = -1
        self.valid_best_f1 = -1

        self.log_path = './result_' + self.touch_name + '.txt'

        self.reset_count()
        global log_f

    def manga4koma_train_sequencial(self):
        pass

    def manga4koma_train(self):
        if self.is_study == False:
            log_f = open(self.log_path, 'w', encoding='utf-8')
            print("class weight : {}".format(w), file=log_f)
        for epoch in range(EPOCHS):
            # === Train ===
            time_start = time.time()
            self.reset_count()

            for x_train, y_train in self.train_loader:

                x_train = Variable(x_train).to(device)
                y_train = Variable(y_train).to(device)

                self.optimizer.zero_grad()

                out = bert_model(x_train)[0][:,0,:]
                y_pred = classifier(out)
                _, predicted = torch.max(y_pred.data, 1)
                self.correct += (predicted == y_train.argmax(1)).sum().item()
                self.total += y_train.size(0)
                loss = self.criterion(y_pred, y_train.argmax(1))
                loss.backward()
                self.optimizer.step()
                self.total_loss += loss.item()

                for i in range(len(predicted)):
                    self.c_mat[torch.max(y_train.data, 1)[1][i]][predicted[i]] += 1
            # ロスの合計を len(train_loader)で割る
            train_mean_loss = self.total_loss / len(self.train_loader)
            train_acc = (self.correct / self.total)
            train_f1 = self.cal_F1(self.c_mat)
            # Historyに追加
            self.train_loss_history.append(train_mean_loss)
            self.train_acc_history.append(train_acc)
            self.train_f1_history.append(train_f1)

            # === Validation ===
            self.reset_count()

            with torch.no_grad():
                for x_valid, y_valid in self.valid_loader:

                    x_valid = Variable(x_valid).to(device)
                    y_valid = Variable(y_valid).to(device)

                    self.optimizer.zero_grad()

                    out = bert_model(x_valid)[0][:,0,:]
                    y_pred = classifier(out)

                    _, predicted = torch.max(y_pred.data, 1)
                    self.correct += (predicted == y_valid.argmax(1)).sum().item()
                    self.total += y_valid.size(0)
                    loss = self.criterion(y_pred, y_valid.argmax(1))
                    self.total_loss += loss.item()

                    for i in range(len(predicted)):
                        self.c_mat[torch.max(y_valid.data, 1)[1][i]][predicted[i]] += 1

                # ロスの合計を len(valid_loader)で割る
                valid_mean_loss = self.total_loss / len(self.valid_loader)
                valid_acc = (self.correct / self.total)
                valid_f1 = self.cal_F1()
                # Historyに追加
                self.valid_loss_history.append(valid_mean_loss)
                self.valid_acc_history.append(valid_acc)
                self.valid_f1_history.append(valid_f1)
                print("---Validation---")
                print(self.c_mat)
                print("Val Acc : %.4f" % valid_acc)
                print("Val F1: {0}".format(valid_f1))

            if valid_f1 > self.valid_best_f1:
                self.valid_best_f1 = valid_f1
                train_best_acc = train_acc
                train_best_f1 = self.train_f1_history[-1]
                val_best_acc = valid_acc

                if self.is_study == False:
                    self.save_model()


            time_finish = time.time() - time_start
            print("====================================")
            print("EPOCH : {0} / {1}".format(epoch + 1, self.EPOCHS))
            print("残り時間 : {0}".format(time_finish * (EPOCHS - epoch)))
            print("VAL_LOSS : {} \nVAL_ACCURACY : {}".format(valid_mean_loss, valid_acc))

            if self.is_study == False:
                print("EPOCH : {0} / {1}".format(epoch + 1, self.EPOCHS), file=log_f)
                print("VAL_LOSS : {} \nVAL_ACCURACY : {}\nVAL_F1 : {}\n".format(valid_mean_loss, valid_acc, valid_f1), file=log_f)

        if self.is_study == False:
            v = Visualizer(self.touch_name, self.train_loss_history, self.train_acc_history, self.train_f1_history,
                           self.valid_loss_history, self.valid_acc_history, self.valid_f1_history)
            v.visualize()

            log_f.close()

        return self.valid_best_f1

    def manga4koma_test(self, test_loader, original_test):
        log_f = open(self.log_path, 'w', encoding='utf-8')
        self.load_model()
        self.reset_count()
        test_index = 0
        label = []
        pred = []
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = Variable(x_test).to(device)
                y_test = Variable(y_test).to(device)

                self.optimizer.zero_grad()

                out = bert_model(x_test)[0][:, 0, :]
                y_pred = classifier(out)

                _, predicted = torch.max(y_pred.data, 1)

                self.correct += (predicted == torch.max(y_test.data, 1)[1]).sum().item()

                self.total += y_test.size(0)

                if (predicted[0] != torch.max(y_test.data, 1)[1][0]):
                    print("# {}話-{}".format(original_test.iloc[test_index].story_main_num,
                                            original_test.iloc[test_index].story_sub_num),
                          file=log_f)
                    print("# 文: {0}\n正解 : {1} , 予測 : {2} / 元クラス : {3}".format(original_test.iloc[test_index].what,
                                                                              torch.max(y_test.data, 1)[1][0],
                                                                              predicted[0],
                                                                              original_test.iloc[test_index].emotion),
                          file=log_f)

                pred.append(predicted[0])
                label.append(torch.max(y_test.data, 1)[1][0])

                test_index = min(len(original_test) - 1, test_index + 1)

            print("------------------------test acc------------------------", file=log_f)
            print("Test Acc : %.4f" % (self.correct / self.total), file=log_f)
            print("correct: {0}, total: {1}".format(self.correct, self.total), file=log_f)
            print("------------------------------------------------", file=log_f)
        d = classification_report([la.tolist() for la in label], [pr.tolist() for pr in pred],
                                  target_names=['喜楽', 'その他'],
                                  output_dict=True)
        df = pd.DataFrame(d)
        print(df, file=log_f)
        log_f.close()

    def save_model(self):
        nlp.save_torch_model(classifier, self.touch_name + '_4_29_classifier_')
        #BertModel.save_pretrained(bert_model, '../models/bert/My_Japanese_transformers/')
        torch.save(bert_model.state_dict(), '../models/bert/My_Japanese_transformers/' + self.touch_name + '_model.bin')
        print('\nbest score updated, Pytorch model was saved!! f1:{}\n'.format(self.valid_best_f1))

    def load_model(self):
        nlp.load_torch_model(self.touch_name + '_4_29_classifier_')
        load_weights = torch.load('../models/bert/My_Japanese_transformers/' + self.touch_name + '_model.bin', map_location={'cuda:0': 'cpu'})
        bert_model.load_state_dict(load_weights)

    def reset_count(self):
        self.total_loss = 0
        self.total = 0
        self.correct = 0
        self.c_mat = np.zeros((2, 2), dtype=int)

    def cal_F1(self, f1_mode=0):
        c_precision = self.c_mat[0][0] / (1e-09 + self.c_mat[0][0] + self.c_mat[0][1])
        c_recall = self.c_mat[0][0] / (1e-09 + self.c_mat[0][0] + self.c_mat[1][0])
        c_f1 = (2 * c_recall * c_precision) / (1e-09 + c_recall + c_precision)
        return c_f1
        #if f1_mode == 0:
        #    return c_f1
        #nc_precision = self.c_mat[1][1] / (1e-09 + self.c_mat[1][1] + self.c_mat[1][0])
        #nc_recall = self.c_mat[1][1] / (1e-09 + self.c_mat[1][1] + self.c_mat[0][1])
        #nc_f1 = (2 * nc_recall * nc_precision) / (1e-09 + nc_recall + nc_precision)
        #if f1_mode == 1:
        #    return nc_f1

        #if f1_mode == 2:
        #    return (c_f1 + nc_f1) / 2


def objective_variable(train_loader, valid_loader, touch_name, is_study):

    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        optimizer = torch.optim.Adam(chain(bert_model.parameters(), classifier.parameters()), lr=lr)
        print("suggest lr = {}".format(lr))
        trainer = Manga4koma_Trainer(optimizer, criterion, train_loader, valid_loader, touch_name, is_study)
        best_valid_f1 = trainer.manga4koma_train()
        error = 1 - best_valid_f1
        return error

    return objective

def main():
    bert_model.to(device)
    classifier.to(device)
    print("experience start device:{}".format(device))

    for touch_name in TOUCH_NAME_ENG:
        print("touch: {}".format(touch_name))
        train_loader, valid_loader, test_loader, test_data_set = get_data_loader(touch_name)
        study = optuna.create_study()
        study.optimize(objective_variable(train_loader, valid_loader, touch_name, is_study=True), n_trials=5)

        print('Number of finished trials: {}'.format(len(study.trials)))

        print('Best trial:')
        trial = study.best_trial

        print('  Value: {}'.format(trial.value))

        print('  Params: ')
        for key, value in trial.params.items():
            print('    {}: {}'.format(key, value))

        optimizer = torch.optim.Adam(chain(bert_model.parameters(), classifier.parameters()), lr=trial.params['lr'])
        ex = Manga4koma_Trainer(optimizer, criterion, train_loader, valid_loader, touch_name, is_study=False)
        ex.manga4koma_train()
        ex.manga4koma_test(test_loader, test_data_set)

if __name__ == '__main__':
    main()