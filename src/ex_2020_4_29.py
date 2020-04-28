from utils import nlp_util
from manga4koma import manga4koma
from my_network.pytorch_mlp import ClassificationNet
import torch
from transformers import BertTokenizer, BertForMaskedLM, BertConfig, BertModel
import time
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
import optuna
from itertools import chain
import numpy as np
from utils import nlp_util as nlp
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
EPOCHS = 10 # エポック数
BATCH_SIZE = 16 # バッチサイズ

config = BertConfig.from_json_file('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json')

bert_model = BertModel.from_pretrained('../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/pytorch_model.bin', config=config)
classifier = ClassificationNet()
criterion = torch.nn.CrossEntropyLoss()
manga_data = manga4koma()

for touch_name in TOUCH_NAME_ENG:
    if touch_name != 'gyagu':
        continue
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

class Manga4koma_Trainer():
    def __init__(self, optimizer, criterion, train_loader, valid_loader):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.train_loss_history = []
        self.train_acc_history = []
        self.train_f1_history = []
        self.valid_loss_history = []
        self.valid_acc_history = []
        self.valid_f1_history = []
        self.valid_best_acc = 0
        self.valid_best_f1 = 0

        self.reset_count()

    def manga4koma_train(self):
        touch_name = 'gyagu'
        print(touch_name + ": train start")
        for epoch in range(EPOCHS):
            # === Train ===

            self.reset_count()

            for x_train, y_train in self.train_loader:

                x_train = Variable(x_train)
                y_train = Variable(y_train)

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

                    x_valid = Variable(x_valid)
                    y_valid = Variable(y_valid)

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
                nlp.save_torch_model(classifier, touch_name + '_4_29_classifier_')
                print('\nbest score updated, Pytorch model was saved!! f1:{}\n'.format(self.valid_best_f1))
                train_best_acc = train_acc
                train_best_f1 = self.train_f1_history[-1]
                val_best_acc = valid_acc

            print("====================================")
            print("EPOCH : {0} / {1}".format(epoch + 1, self.EPOCHS))
            print("VAL_LOSS : {} \nVAL_ACCURACY : {}".format(valid_mean_loss, valid_acc))

        return self.valid_best_f1

    def reset_count(self):
        self.total_loss = 0
        self.total = 0
        self.correct = 0
        self.c_mat = np.zeros((2, 2), dtype=int)

    def cal_F1(self, f1_mode=0):
        c_precision = self.c_mat[0][0] / (1e-09 + self.c_mat[0][0] + self.c_mat[0][1])
        c_recall = self.c_mat[0][0] / (1e-09 + self.c_mat[0][0] + self.c_mat[1][0])
        c_f1 = (2 * c_recall * c_precision) / (1e-09 + c_recall + c_precision)
        if f1_mode == 0:
            return c_f1
        nc_precision = self.c_mat[1][1] / (1e-09 + self.c_mat[1][1] + self.c_mat[1][0])
        nc_recall = self.c_mat[1][1] / (1e-09 + self.c_mat[1][1] + self.c_mat[0][1])
        nc_f1 = (2 * nc_recall * nc_precision) / (1e-09 + nc_recall + nc_precision)
        if f1_mode == 1:
            return nc_f1

        if f1_mode == 2:
            return (c_f1 + nc_f1) / 2



def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    optimizer = torch.optim.Adam(chain(bert_model.parameters(), classifier.parameters()), lr=lr)
    trainer = Manga4koma_Trainer(optimizer, criterion, train_loader, valid_loader)
    best_valid_f1 = trainer.manga4koma_train()
    error = 1 - best_valid_f1
    return error



def main():
    print("experience start")
    study = optuna.create_study()
    study.optimize(objective, n_trials=100, timeout=600)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

if __name__ == '__main__':
    main()