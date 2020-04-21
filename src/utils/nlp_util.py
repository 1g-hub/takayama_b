# 自然言語処理でよく使うライブラリ
import numpy as np
import matplotlib.pyplot as plt
import MeCab
import torch
from sklearn.metrics import accuracy_score
from sklearn.utils.extmath import randomized_svd
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import cloudpickle

# textの事前処理(corpus,辞書{単語:id　と id:単語})
def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')

    word_to_id = {}
    id_to_word = {}

    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

def create_word_dic(texts):
    word_to_id = {}
    id_to_word = {}

    word_to_id["<pad>"] = 0
    id_to_word[0] = "<pad>"

    corpus = []

    for text in texts:
        for word in text:
            if word not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[word] = new_id
                id_to_word[new_id] = word

        corpus.append([word_to_id[w] for w in text])
    return np.array(corpus), word_to_id, id_to_word

# 共起行列の生成
def create_co_matrix(corpus, vocab_size, window_size=1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range (1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


# ベクトル同士のコサイン類似度
def cos_similarity(x, y, eps=1e-8):
    nx = x / (np.sqrt(np.sum(x**2)) + eps)  # 正規化
    ny = y / (np.sqrt(np.sum(y**2)) + eps)  # 正規化
    return np.dot(nx, ny)


# コサイン類似度のランキング上位取り出し
def most_cos_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    # クエリ取り出し
    if query not in word_to_id:
        print("%s is not found." % query)
        return

    print("\n[query] " + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]

    # コサイン類似度の導出
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)

    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)

    # 高い順にtop個出力
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(" %s: %s" % (id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


# Positive PMI(正の相互情報量)の行列 (共起行列 -> positive_pmi行列)
def positive_pmi(C, verbose=False, eps=1e-8):
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print("%.1f%% done" % (100*cnt/total))

    return M


# 次元削減
def svd(p_pmi):
    U, S, V = np.linalg.svd(p_pmi)
    # U:密なベクトル表現を格納
    return U, S, V


# 高速次元削減
def truncated_svd(p_pmi, word_vec_size=100, _n_iter=5, _random_state=None):
    U, S, V = randomized_svd(p_pmi, n_components=word_vec_size, n_iter=_n_iter, random_state=_random_state)
    return U, S, V


# MeCab分かち書き
def split2words_mecab(texts):
    parser = lambda x: MeCab.Tagger(' -Owakati').parse(x).strip().split()
    res = []
    for text in texts:
        res.append(parser(text))
    return np.array(res)

def jumanpp_wakati2array(texts):
    res = []
    for text in texts:
        res.append(text.split())
    return np.array(res)


# doc2vec用のデータ整形
def create_tagged_document(texts):
    # texts:分かち書きされた文書のリスト

    # １文書ずつ、単語に分割してリストに入れていく[([単語1,単語2,単語3],文書id),...]こんなイメージ
    # words：文書に含まれる単語のリスト（単語の重複あり）
    # tags：文書の識別子（リストで指定．1つの文書に複数のタグを付与できる）
    trainings = [TaggedDocument(words=text, tags=[i]) for i, text in enumerate(texts)]

    #print(trainings[0])

    return trainings


# doc2vec 訓練
def train_doc2vec(train_data, model_name, vector_size=300, epochs=30, window=8, min_count=5, alpha=0.05, workers=4, retrain=True):
    """
    dm: 1の時はPV-DMを学習, それ以外の場合はPV-DBOW.
    vector_size: 分散表現の次元数.
    window: N-gramのNのサイズ.
    alpha: 学習率
    min_alpha: 最低学習率. 学習率はこのモデルではこれ以上下がらない.
    min_count: 出現回数がmin_count以上のものだけ単語としてとる.
    sample: downsampling, つまり接続詞等の非常に頻繁に出現する単語を無視する確率である.
    workers: 並列実行数
    """


    # モデルのセーブパス
    save_path = 'model/' + model_name + '_doc2vec.model'

    if os.path.exists(save_path) and retrain is not True:
        m = Doc2Vec.load(save_path)
        return m
    else:
        m = Doc2Vec(documents=train_data, dm=1, vector_size=vector_size, epochs=epochs, window=window, min_count=min_count, alpha=alpha, workers=workers)
        # モデルのセーブ
        m.save(save_path)
    return m


def load_d2v_model(model_name):
    save_path = 'model/' + model_name + '_doc2vec.model'
    if os.path.exists(save_path):
        m = Doc2Vec.load(save_path)
        return m
    else:
        print("<<ERROR>> model is not exist.")


def train_torch_net(model, train_loader, model_name, retrain=False):
    # モデルのセーブパス
    save_path = 'model/' + model_name + '.pkl'

    if os.path.exists(save_path) and retrain is not True:
        m = load_torch_model(model_name)
        return m
    else:

        model.train()

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        loss_history = []
        acc_history = []
        for epoch in range(1000):
            total_loss = 0
            correct = 0
            total = 0
            for x_train, y_train in train_loader:
                x_train = Variable(x_train)
                y_train = Variable(y_train)
                optimizer.zero_grad()
                y_pred = model(x_train)

                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == y_train.argmax(1)).sum().item()
                total += y_train.size(0)

                loss = criterion(y_pred, y_train)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            loss_history.append(total_loss)
            acc_history.append((correct / total))
            if (epoch + 1) % 100 == 0:
                print(epoch + 1, total_loss)
        plt.plot(loss_history, label="loss")
        plt.plot(acc_history, label="accuracy")
        plt.xlabel("epoch")
        plt.title(model_name + "_train")
        plt.legend()
        plt.savefig("result/" + model_name + "_train.png")
        plt.show()

        save_torch_model(model, model_name)
        return model


def eval_torch_net(model, model_name, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            outputs = model(x_test)
            _, predicted = torch.max(outputs.data, 1)
            print(y_test.size(0))
            print(predicted)
            print(y_test.argmax(1))
            correct += (predicted == y_test.argmax(1)).sum().item()
            total += y_test.size(0)
    with open('result/' + model_name + '_result.txt', 'wt') as fout:
        print("------------------------------------------------", file=fout)
        print("Test Acc : %.4f" % (correct/total), file=fout)
        print("correct: {0}, total: {1}".format(correct, total), file=fout)
        print("------------------------------------------------", file=fout)


def save_torch_model(model, model_name):
    with open('model/' + model_name + '.pkl', 'wb') as f:
        cloudpickle.dump(model, f)


def load_torch_model(model_name):
    with open('model/' + model_name + '.pkl', 'rb') as f:
        model = cloudpickle.load(f)
    return model