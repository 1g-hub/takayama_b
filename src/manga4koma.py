# coding: utf-8
from collections import defaultdict

import numpy as np
import pandas as pd

import pickle
import os
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import torch

def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
CONV_TOUCH = {"gyagu": "ギャグタッチ", "shoujo": "少女漫画タッチ", "shounen": "少年漫画タッチ", "seinen": "青年漫画タッチ", "moe": "萌え系タッチ"}
EMB_MODE = ["d2v", "bert"]
EMB_DIM = {"d2v": 300, "bert": 768}
CONV_EMO = {"ニュートラル": 'neutral', "驚愕": 'kyougaku', "喜楽": 'kiraku', "恐怖": 'kyouhu', "悲哀": 'hiai', "憤怒": 'hunnu', "嫌悪": 'keno'}
SEQ_LEN = [2, 3, 4, 5, 6]

class manga4koma():
    def __init__(self, to_zero_pad=False, to_sequential=False, seq_len=3, mode='kyoto'):
        self.TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]
        self.CONV_TOUCH = {"gyagu": "ギャグタッチ", "shoujo": "少女漫画タッチ", "shounen": "少年漫画タッチ", "seinen": "青年漫画タッチ", "moe": "萌え系タッチ"}
        self.EMB_MODE = ["d2v", "bert"]
        self.EMB_DIM = {"d2v": 300, "bert": 768}
        self.CONV_EMO = {"ニュートラル": 'neutral', "驚愕": 'kyougaku', "喜楽": 'kiraku', "恐怖": 'kyouhu', "悲哀": 'hiai', "憤怒": 'hunnu', "嫌悪": 'keno'}
        self.SEQ_LEN = [2, 3, 4, 5, 6]

        # 各4コマの最初には台詞がないので, '[PAD]' というテキストのセリフを付加するかどうか
        self.to_zero_pad = to_zero_pad
        self.to_sequential = to_sequential
        self.seq_len = seq_len

        # mode : 'kyoto' 京大BERT, 'hotto' hottoSNS-BERT
        self.mode = mode

        self.change_mode(new_mode=self.mode)

    def change_mode(self, new_mode='hotto'):
        self.mode = new_mode

        if self.mode == 'kyoto':
            self.bert_tokenizer = BertTokenizer(
                '../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt',
                do_lower_case=False, do_basic_tokenize=False,
                config='../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/tokenizer_config.json')
        elif self.mode == 'hotto':
            self.bert_tokenizer = BertTokenizer("../models/bert/hottoSNS-bert-pytorch/vocab.txt", do_lower_case=False,
                                                tokenize_chinese_chars=False, unk_token='<unk>', pad_token='<pad>',
                                                init_inputs=[])
        self.__set_data()
        self.__tokenize(seq_len=self.seq_len)

    def __set_data(self):
        self.data = defaultdict(pd.DataFrame)
        self.original_data = defaultdict(pd.DataFrame)
        #self.bert_tokenized = defaultdict(dict)
        #self.tokenized = defaultdict(list)

        for touch_name in self.TOUCH_NAME_ENG:
            self.data[touch_name] = pd.read_csv(
                '../new_dataset/' + touch_name + '_modified_aug_add_sub.csv',
                index_col=0,
                dtype={'original': bool,'inner': bool, 'kakimoji': bool, 'self_anotated': bool, 'alter_emotion': str},
                usecols=lambda x: x is not 'index'
            )

            # koma_vec列の追加
            self.data[touch_name] = self.data[touch_name].assign(koma_vec = None)
            self.original_data[touch_name] = self.data[touch_name][self.data[touch_name].original]

            if self.mode == 'kyoto':
                self.data[touch_name].wakati = [w.split(' ') for w in self.data[touch_name].subword.tolist()]
                self.original_data[touch_name].wakati = [w.split(' ') for w in
                                                         self.original_data[touch_name].subword.tolist()]
            elif self.mode == 'hotto':
                self.data[touch_name].wakati = [w.split(' ') for w in self.data[touch_name].wakati_sp.tolist()]
                self.original_data[touch_name].wakati = [w.split(' ') for w in
                                                         self.original_data[touch_name].wakati_sp.tolist()]


    def __tokenize(self, seq_len=3):
        for touch_name in self.TOUCH_NAME_ENG:

            # koma_vec の代入
            for i, v in self.data[touch_name].iterrows():
                if os.path.exists('../dataset/4koma_vec/' + touch_name + '_' + str(v['story_main_num']+1).zfill(3) + '_' + str(v['story_sub_num']+1) + '-' + str(v['koma']+1) + '.pkl'):
                    self.data[touch_name].at[i, 'koma_vec'] = pickle_load('../dataset/4koma_vec/' + touch_name + '_' + str(v['story_main_num']+1).zfill(3) + '_' + str(v['story_sub_num']+1) + '-' + str(v['koma']+1) + '.pkl')
                else:
                    self.data[touch_name].at[i, 'koma_vec'] = pickle_load('../dataset/4koma_vec/pad.pkl')

            if self.to_zero_pad:
                self.zero_padding(touch_name, seq_len)
            s = self.data[touch_name].wakati.tolist()
            #self.new_data[touch_name] = self.data[touch_name].assign(to)
            self.data[touch_name]['tokenized'] = s

            res_encode = self.bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)

            # self.data[touch_name]['bert_tokenized'] = [b for b in bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)]

            self.data[touch_name]['input_ids'] = \
                [b for b in res_encode['input_ids']]
            self.data[touch_name]['token_type_ids'] = \
                [b for b in res_encode['token_type_ids']]
            self.data[touch_name]['attention_mask'] = \
                [b for b in res_encode['attention_mask']]

            # res_encode = bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)
            # self.data[touch_name]['bert_tokenized'] = []
            #
            # for i in range(len(res_encode['input_ids'])):
            #     self.data[touch_name]['bert_tokenized'].append({'input_ids': res_encode['input_ids'][i], 'token_type_ids': res_encode['token_type_ids'][i],
            #               'attention_mask': res_encode['attention_mask'][i]})


            if self.to_sequential:
                self.to_seq(touch_name, seq_len)
        del s, res_encode


    def zero_padding(self, touch_name, seq_len=3):
        data_columns = ['id', 'original', 'story_main_num', 'story_sub_num', 'koma', 'who', 'inner', 'speaker', 'what',
                        'wakati', 'emotion', 'kakimoji', 'self_anotated', 'alter_emotion', 'wakati_sp']
        data = dict.fromkeys(data_columns)
        padding_serif = '[PAD]' if self.mode == 'kyoto' else '<pad>'
        data['original'] = True
        data['koma'] = 0
        data['emotion'] = 'None'
        data['what'] = padding_serif
        data['koma_vec'] = pickle_load('../dataset/4koma_vec/pad.pkl')
        # data['wakati'] = np.array([np.zeros((1, 300))]).reshape(300)
        data['wakati'] = [padding_serif]
        story_main_max = self.data[touch_name].story_main_num.max()
        story_main_min = self.data[touch_name].story_main_num.min()
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

            df = self.data[touch_name][(self.data[touch_name].story_main_num == i) & (self.data[touch_name].story_sub_num == 0)]
            df.id = df.id + (seq_len - 1) * cnt
            sep.append(df)
            now_id = df.id.max() + 1

            data['story_sub_num'] = 1

            for seq in range(0, seq_len - 1):
                data['id'] = now_id
                now_id += 1
                sep.append(pd.DataFrame.from_dict([data]))

            cnt += 1

            df = self.data[touch_name][(self.data[touch_name].story_main_num == i) & (self.data[touch_name].story_sub_num == 1)]
            df.id = df.id + (seq_len - 1) * cnt
            sep.append(df)
            now_id = df.id.max() + 1

        new_data_set = pd.concat([s for s in sep], sort=False)
        self.data[touch_name] = new_data_set.reset_index(drop=True)

    def to_seq(self, touch_name, seq_len=3):
        #bert_tokenizeをいじる
        self.seq_data = defaultdict(dict)
        f_trains = self.data[touch_name][self.data[touch_name].original]

        x_input = []
        x_type = []
        x_attn = []
        y_koma = []

        for f_index, f_train_data in f_trains.iterrows():

            front_in = pd.DataFrame([{field: None for field in self.data[touch_name].columns.values}])

            give_up = False

            for seq in range(seq_len - 1):
                next = f_trains[(f_trains.id - seq == f_train_data.id) & (f_trains.story_sub_num == f_train_data.story_sub_num)]
                if len(next) == 0:
                    give_up = True
                    break
                else:
                    next = next.iloc[0]
                front_in = front_in.append(next)

            if give_up:
                continue

            third_ins = self.data[touch_name][(self.data[touch_name].id - 1 == front_in.iloc[-1].id) & (self.data[touch_name].story_sub_num == front_in.iloc[-1].story_sub_num)]
            # print(front_in.iloc[1].what + " " + front_in.iloc[2].what)
            for t_index, third_in in third_ins.iterrows():

                x_input.append(np.stack([*front_in.tail(seq_len-1).input_ids, third_in.input_ids], axis=0))
                x_type.append(np.stack([*front_in.tail(seq_len-1).token_type_ids, third_in.token_type_ids], axis=0))
                x_attn.append(np.stack([*front_in.tail(seq_len-1).attention_mask, third_in.attention_mask], axis=0))

                y_koma.append(np.stack([*front_in.tail(seq_len-1).koma_vec, third_in.koma_vec], axis=0))

        if self.mode == 'kyoto':
            self.data[touch_name] = self.data[touch_name][self.data[touch_name]['what'] != '[PAD]']
        else:
            self.data[touch_name] = self.data[touch_name][self.data[touch_name]['what'] != '<pad>']
        self.data[touch_name] = self.data[touch_name].reset_index(drop=True)
        self.data[touch_name].input_ids = x_input
        self.data[touch_name].token_type_ids = x_type
        self.data[touch_name].attention_mask = x_attn
        self.data[touch_name].koma_vec = y_koma
        #x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=val_size, random_state=random_seed, shuffle=shuffle)

    def make_5touch_concat(self):
        data_columns = ['id', 'original', 'story_main_num', 'story_sub_num', 'koma', 'who', 'inner', 'speaker', 'what',
                        'wakati', 'emotion', 'kakimoji', 'self_anotated', 'alter_emotion', 'wakati_sp', 'touch']
        data = pd.DataFrame(columns=data_columns)

        for touch_name in self.TOUCH_NAME_ENG:
            touch_data = self.data[touch_name]
            touch_data = touch_data.assign(touch = touch_name)
            data = pd.concat([data, touch_data])

        s = data.wakati.tolist()
        # self.new_data[touch_name] = self.data[touch_name].assign(to)
        data['tokenized'] = s

        res_encode = self.bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True,
                                                           is_pretokenized=True)

        # self.data[touch_name]['bert_tokenized'] = [b for b in bert_tokenizer.batch_encode_plus(s, pad_to_max_length=True, add_special_tokens=True, is_pretokenized=True)]

        data['input_ids'] = \
            [b for b in res_encode['input_ids']]
        data['token_type_ids'] = \
            [b for b in res_encode['token_type_ids']]
        data['attention_mask'] = \
            [b for b in res_encode['attention_mask']]

        return data



# amanga4koma = manga4koma(to_zero_pad=False, to_sequential=False, seq_len=3, mode='hotto')
# a = amanga4koma.make_5touch_concat()
# print(a)
# print(a.emotion)
#
# b = a[a.touch == 'shounen']
# print(b)
# print(b.emotion)
#
# b = a[a.touch == 'shoujo']
# print(b)
# print(b.emotion)
# print(a.touch)
# print(amanga4koma.data['gyagu'].wakati[0])
# print(amanga4koma.data['gyagu'].wakati_sp[0])
# print(amanga4koma.data['gyagu'].wakati[2])
# print(amanga4koma.data['gyagu'].wakati_sp[2])
# print(amanga4koma.data['gyagu'].input_ids[0])
# print(amanga4koma.data['gyagu'].input_ids[2])
# amanga4koma.change_mode('kyoto')
# print(amanga4koma.data['gyagu'].wakati[0])
# print(amanga4koma.data['gyagu'].wakati_sp[0])
# print(amanga4koma.data['gyagu'].wakati[2])
# print(amanga4koma.data['gyagu'].wakati_sp[2])
# print(amanga4koma.data['gyagu'].input_ids[0])
# print(amanga4koma.data['gyagu'].input_ids[2])

# for i in amanga4koma.data['gyagu'].input_ids:
#     print(i)
#
# for i in amanga4koma.data['gyagu'].what:
#     print(i)
# print(amanga4koma.data['gyagu'].koma_vec[0])
# print(amanga4koma.data['moe'].koma_vec[0])
# print(amanga4koma.data['moe'].koma_vec[7562])
# print(amanga4koma.data['moe'].what)
# print(amanga4koma.data['gyagu'].input_ids)
# print(amanga4koma.data['gyagu'].token_type_ids)
# print(amanga4koma.data['gyagu'].attention_mask)
# print(amanga4koma.data['gyagu'].bert_tokenized[0])
#print(bert_tokenizer.convert_ids_to_tokens(amanga4koma.data['gyagu'].bert_tokenized[7862]))
#print(amanga4koma.data['gyagu'].wakati)

#a = np.stack([*amanga4koma.data['gyagu'].bert_tokenized],axis=0)
#print(a)
#embedding = Embbedding(amanga4koma.tokenized, "../models/d2v_manga109.model", mode='d2v', word_base=True)
#print(embedding.embed['gyagu'].size)
#print(embedding.embed['gyagu'])
#print(amanga4koma.data['gyagu'].what)



# input_ids = torch.tensor(amanga4koma.data['gyagu']['input_ids'])
# for i in range(10):
#     print(bert_tokenizer.convert_ids_to_tokens(input_ids[i].tolist()))
#
# amanga4koma = manga4koma(to_zero_pad=False, to_sub_word=False, to_sequential=False, seq_len=3)
#
# input_ids = torch.tensor(amanga4koma.data['gyagu']['input_ids'])
# for i in range(10):
#     print(bert_tokenizer.convert_ids_to_tokens(input_ids[i].tolist()))


