# coding: utf-8
import pandas as pd
from msvcrt import getch
import random
import os


file_path = './data.csv'

# ジャンル別問題数
n = 3

class Koujutsu():
    def __init__(self, file_path, g_min=0, g_max=10):
        self.data = load_data(file_path)
        self.g_min = g_min
        self.g_max = g_max
        self.make_question()

    def make_question(self):
        self.q_data = pd.DataFrame(columns=['group','question'])
        turn_list = list(range(self.g_min + 1, self.g_max + 1))
        random.shuffle(turn_list)
        turn_list.append(0)
        turn_list.reverse()
        for g in turn_list:

            sample_num = min(len(self.data[self.data.group == g]), n)
            self.q_data = pd.concat([self.q_data, self.data[self.data.group == g].sample(sample_num)])
        self.q_data = self.q_data.reset_index(drop=True)


def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0, encoding="shift-jis")
    return data

def show_test(data):
    now_index = 0
    q_len = len(data)
    print("試験開始 (Enter キーでスタート)")
    while True:
        key = ord(getch())
        # Esc
        if key == 27:
            break
        # Enter
        elif key == 13:
            if now_index == q_len:
                os.system('cls')
                print("\n試験終了\n")
                break
            os.system('cls')
            print('[' + str(now_index + 1) + '/' + str(q_len) + ']' + ' :\n' + data.question[now_index])
            now_index += 1

def main():
    koujutsu = Koujutsu(file_path)
    show_test(koujutsu.q_data)



if __name__ == '__main__':
    main()
