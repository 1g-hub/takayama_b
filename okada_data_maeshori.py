# coding: utf-8
import re
import pandas as pd
from pandas.io.json import json_normalize
from pyknp import Juman

"""
########
これを動かすときのみ、
インタープリターをUbuntuにしてください！(Juman++)
########
"""


def load_text(file_path):
    text = []
    with open(file_path, mode='r', encoding='shift_jis') as f:
        for line in f:
            # 全角があれば半角に変更, 改行コードは取り除く
            line = line.translate(str.maketrans({chr(0xFF01 + i): chr(0x21 + i) for i in range(94)})).rstrip(
                '\n').replace('\u3000', ' ')
            if line != '':
                text.append(line)
    return text


def regex_text(text):
    data_columns = ['id', 'original', 'story_main_num', 'story_sub_num', 'koma', 'who', 'inner', 'speaker', 'what', 'wakati',
                    'emotion']  # 将来的には 'to' も入れる
    characters = ['A', 'B', 'Bらしき後ろ姿の別人女性', 'アイドル']
    data_set = []

    patns = {'start_pos': '■', 'start_koma': '\[(\d)\]', 'emotion': '\(感情:(.+?)\)[^(「]', 'inner_voice': '[^「」]+心の声',
             'kaisou': '[^「」]+回想', 'speak_other': '\{(.+?)\}', 'what_in': '\(([^(感情:)].+?)\)', 'what_out': '「(.+?)」'}
    # TODO A心の声(感情:)(ベタだなあ) の扱いはどうするか?
    # TODO 少女マンガタッチ B書き文字（感情：喜楽）：いいよー の扱い(今は使ってない)
    id = -1
    story_main_num = -1
    story_sub_num = -1

    for sentence in text:
        sentence = sentence.replace('(', ' (').replace('「', ' 「')
        data = dict.fromkeys(data_columns)

        # マンガのstory_numとコマ数
        if re.match(patns['start_pos'], sentence):
            story_sub_num = (story_sub_num + 1) % 2
            if story_sub_num == 0:
                story_main_num += 1
        elif re.search(patns['start_koma'], sentence):
            koma = int(re.findall(patns['start_koma'], sentence)[0]) - 1

        # 文章からデータを取り出す
        if re.search(patns['emotion'], sentence):
            # 使うデータ => 感情ラベルあり && セリフ付き(心の中での発言 or 話し)
            sentence = sentence.replace('ニュートラル (無感情)', 'ニュートラル')  # 表記ゆれ対策
            if re.search(patns['what_in'], sentence) or re.search(patns['what_out'], sentence):
                id = max(0, id + 1)

                data['id'] = id
                data['emotion'] = re.findall(patns['emotion'], sentence, re.S)[0]
                data['story_main_num'] = story_main_num
                data['story_sub_num'] = story_sub_num
                data['koma'] = koma
                data['inner'] = True if (re.search(patns['inner_voice'], sentence) or (
                            re.search(patns['kaisou'], sentence) and re.search(patns['what_in'], sentence))) else False

                for w in re.split('[、\s]', sentence):
                    if w in characters:
                        data['who'] = w
                        break
                    else:
                        for t in w:
                            if t in characters:
                                data['who'] = t
                                break

                if data['inner']:
                    res = re.findall(patns['what_in'], sentence, re.S)
                    data['what'] = ' '.join(res)
                    if re.search(patns['speak_other'], sentence, re.S):
                        data['speaker'] = str(re.findall(patns['speak_other'], sentence, re.S)[0])
                    else:
                        data['speaker'] = data['who']
                else:
                    res = re.findall(patns['what_out'], sentence, re.S)
                    data['what'] = ' '.join(res)
                    data['speaker'] = data['who']

                data['wakati'] = create_wakati(data['what'])
                data['original'] = True
                data_set.append(data)

    return json_normalize(data=data_set)


def create_wakati(text):
    result = Juman(jumanpp=True).analysis(text)
    wakalist = [mrph.midasi for mrph in result.mrph_list()]
    dame = ["", " ", "　", "、", "。"]
    wakati = ' '.join(waka for waka in wakalist if waka not in dame)
    return wakati


def main():
    touch_name = ["ギャグタッチ", "少女漫画タッチ", "少年漫画タッチ", "青年漫画タッチ", "萌え系タッチ"]
    file_append = '_takayama.txt'

    for touch in touch_name:
        file_path = touch + file_append
        text = load_text(file_path)
        res = regex_text(text)

        print(res)
        res.to_csv('../dataset/' + touch + '.csv', encoding='utf_8_sig')


if __name__ == '__main__':
    main()