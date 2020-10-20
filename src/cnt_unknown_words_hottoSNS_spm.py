# coding: utf-8
from collections import defaultdict
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM, BertConfig
import sentencepiece as sp
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

file = "../models/bert/hottoSNS-bert-pytorch/vocab.txt"
bert_words = load_text(file)
print(bert_words)

for b in bert_words:
    if '#' in b:
        print(b)

# GLOBAL 変数
TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]

log = open('./cnt_unkwords_hottoSNS_spm.txt',mode='w', encoding='utf-8')

bert_tokenizer = BertTokenizer("../models/bert/hottoSNS-bert-pytorch/vocab.txt",config='../models/bert/hottoSNS-bert-pytorch/tokenizer_config.json', unk_token='<unk>')

s = 'ゆめさんが、ファボしてくるあたり、世代だなって思いました(   ̇- ̇  )笑'
print(bert_tokenizer.tokenize(s))
sp_tokenizer = sp.SentencePieceProcessor()
sp_tokenizer.load("../models/bert/hottoSNS-bert-pytorch/tokenizer_spm_32K.model")
#s = ' '.join(sp_tokenizer.EncodeAsPieces(s))
s = (sp_tokenizer.EncodeAsPieces(s))
print(s)
print(bert_tokenizer.convert_tokens_to_ids(s))
print(bert_tokenizer.convert_tokens_to_ids(['赤城','一ノ瀬志希']))
print(bert_tokenizer.convert_tokens_to_ids(['[UNK]','<unk>']))

s = 'bさん、苦手そうだよな…'
s = (sp_tokenizer.EncodeAsPieces(s))
print(s)
print(bert_tokenizer.convert_tokens_to_ids(s))
# for touch_name in TOUCH_NAME_ENG:
#     unk_words = []
#     known_words = []
#     data = pd.read_csv(
#         '../dataset/' + touch_name + '_augmentation.csv',
#         index_col=0,
#         dtype={'original': bool},
#         usecols=lambda x: x is not 'index'
#     )
#     original_data = data[data.original]
#
#     # data.what = [w.split(' ') for w in data.what.tolist()]
#     # original_data.what = [w.split(' ') for w in original_data.what.tolist()]
#     #
#     # for s in original_data.what:
#     #     print("what:")
#     #     print(s[0])
#     #     wakati = bert_tokenizer.tokenize(s[0])
#     #     print("wakati:")
#     #     print(wakati)
#     sp_tokenizer = sp.SentencePieceProcessor()
#     sp_tokenizer.load("../models/bert/hottoSNS-bert-pytorch/tokenizer_spm_32K.model")
#
#     data.wakati = [sp_tokenizer.EncodeAsPieces(w) for w in data.what.tolist()]
#     original_data.wakati = [sp_tokenizer.EncodeAsPieces(w) for w in original_data.what.tolist()]
#
#     for wakati in original_data.wakati:
#         print(wakati)
#         all = []
#         for word_before in wakati:
#             word_after = bert_tokenizer.tokenize(word_before)
#             all.append(' '.join(word_after))
#
#             for w in word_after:
#                 if w == '[UNK]':
#                     word = ''.join(word_before)
#                     print(word)
#                     if word not in unk_words:
#                         unk_words.append(word)
#                 else:
#                     if w not in known_words:
#                         known_words.append(w)
#
#         print(all)
#         print("")
#
#     print("タッチ : {}".format(touch_name), file=log)
#     print("総単語数 : {}".format(len(unk_words) + len(known_words)), file=log)
#     print("未知語数 : {}".format(len(unk_words)), file=log)
#     print("未知語率 : {}".format( len(unk_words) / (len(unk_words) + len(known_words)) ) , file=log)
#     print(unk_words, file=log)
#
#     unk_words = []
#     known_words = []
#
#     for wakati in data.wakati:
#         print(wakati)
#         all = []
#         for word_before in wakati:
#             word_after = bert_tokenizer.tokenize(word_before)
#             all.append(' '.join(word_after))
#
#             for w in word_after:
#                 if w == '[UNK]':
#                     word = ''.join(word_before)
#                     print(word)
#                     if word not in unk_words:
#                         unk_words.append(word)
#                 else:
#                     if w not in known_words:
#                         known_words.append(w)
#
#         print(all)
#         print("")
#     print("タッチ : {}".format(touch_name), file=log)
#     print("Augmentated総単語数 : {}".format(len(unk_words) + len(known_words)), file=log)
#     print("Augmentated未知語数 : {}".format(len(unk_words)), file=log)
#     print("Augmentated未知語率 : {}".format( len(unk_words) / (len(unk_words) + len(known_words)) ), file=log )
#     print(unk_words, file=log)
#

log.close()


