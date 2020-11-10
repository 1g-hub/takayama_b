# coding: utf-8
from collections import defaultdict
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
def load_text(file_path):
    text = []
    with open(file_path, mode='r', encoding='utf_8') as f:
        for line in f:
            # 全角があれば半角に変更, 改行コードは取り除く
            line = line.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})).rstrip(
                '\n').replace('\u3000', ' ')
            if line != '':
                text.append(line)
    return text

file = '../models/bert/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/vocab.txt'

bert_words = load_text(file)
print(bert_words)


# GLOBAL 変数
TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]

log = open('./cnt_unkwords_hottoSNS.txt',mode='w', encoding='utf-8')

bert_tokenizer = BertTokenizer("../models/bert/hottoSNS-bert-pytorch/vocab.txt",do_lower_case=False, do_basic_tokenize=False)

for touch_name in TOUCH_NAME_ENG:
    unk_words = []
    known_words = []
    data = pd.read_csv(
        '../dataset/' + touch_name + '_augmentation.csv',
        index_col=0,
        dtype={'original': bool},
        usecols=lambda x: x is not 'index'
    )
    original_data = data[data.original]

    data.wakati = [w.split(' ') for w in data.wakati.tolist()]
    original_data.wakati = [w.split(' ') for w in original_data.wakati.tolist()]

    for wakati in original_data.wakati:
        for word_before in wakati:
            word_after = bert_tokenizer.tokenize(word_before)
            # print(word_after)
            # print(word_before)

            # w_ori = ''.join(word_before)
            # # [UNK] を元に戻す.
            # w_a = ''.join(word_after).replace('##','').split('[UNK]')
            # print(w_a)
            # unk_num = len(w_a) - 1
            # for w in w_a:
            #     w_ori.replace(w,' ')
            # w_ori.split(' ')
            # print(w_ori)
            for word in word_after:
                if word == '[UNK]':
                    word = ''.join(word_before)
                    print(word)
                w_id = bert_tokenizer.convert_tokens_to_ids(word)
                if word in known_words:
                    continue
                else:
                    if w_id != 1:
                        if word == 'A':
                            print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        known_words.append(word)
                    else:
                        if word not in unk_words:
                            unk_words.append(word)
                    # if word in bert_words:
                    #     if word == 'A':
                    #         print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    #     known_words.append(word)
                    # else:
                    #     if word not in unk_words:
                    #         unk_words.append(word)
    print("タッチ : {}".format(touch_name), file=log)
    print("総単語数 : {}".format(len(unk_words) + len(known_words)), file=log)
    print("未知語数 : {}".format(len(unk_words)), file=log)
    print("未知語率 : {}".format( len(unk_words) / (len(unk_words) + len(known_words)) ) , file=log)
    print(unk_words, file=log)

    unk_words = []
    known_words = []

    # for wakati in data.wakati:
    #     for word_before in wakati:
    #         word_after = bert_tokenizer.tokenize(word_before)
    #         # print(word_after)
    #         # print(word_before)
    #
    #         # w_ori = ''.join(word_before)
    #         # # [UNK] を元に戻す.
    #         # w_a = ''.join(word_after).replace('##','').split('[UNK]')
    #         # print(w_a)
    #         # unk_num = len(w_a) - 1
    #         # for w in w_a:
    #         #     w_ori.replace(w,' ')
    #         # w_ori.split(' ')
    #         # print(w_ori)
    #         for word in word_after:
    #             print(word)
    #             if word == '[UNK]':
    #                 word = ''.join(word_before)
    #                 #print(word)
    #             w_id = bert_tokenizer.convert_tokens_to_ids(word)
    #             if word in known_words:
    #                 continue
    #             else:
    #                 if w_id != 1:
    #                     if word == 'A':
    #                         print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #                     known_words.append(word)
    #                 else:
    #                     if word not in unk_words:
    #                         unk_words.append(word)
    #             # if word in known_words:
    #             #     continue
    #             # else:
    #             #     if word in bert_words:
    #             #         known_words.append(word)
    #             #     else:
    #             #         if word not in unk_words:
    #             #             unk_words.append(word)
    # print("タッチ : {}".format(touch_name), file=log)
    # print("Augmentated総単語数 : {}".format(len(unk_words) + len(known_words)), file=log)
    # print("Augmentated未知語数 : {}".format(len(unk_words)), file=log)
    # print("Augmentated未知語率 : {}".format( len(unk_words) / (len(unk_words) + len(known_words)) ), file=log )
    # print(unk_words, file=log)

log.close()


