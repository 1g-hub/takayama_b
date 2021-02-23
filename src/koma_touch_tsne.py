# coding: utf-8
import seaborn as sb
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE
from manga4koma import manga4koma
amanga = manga4koma(to_zero_pad=False, to_sequential=False, seq_len=3, mode='hotto').make_5touch_concat()
EMO = ["ニュートラル", "驚愕", "喜楽", "恐怖", "悲哀", "憤怒", "嫌悪"]
TOUCH_NAME_ENG = ["gyagu", "shoujo", "shounen", "seinen", "moe"]

manga = amanga[amanga.original]
n_components = [2]
perplexity = [10, 20, 30, 50, 70, 100, 150]

X = np.array(manga.koma_vec.values.tolist())
print(X)
y = np.array(manga.touch.values.tolist())
y_items = TOUCH_NAME_ENG

print(y_items)

for n in n_components:
    for p in perplexity:
        fig, ax = plt.subplots(figsize=(10, 10))
        tsne = TSNE(n_components=n, init='random', random_state=25, perplexity=p)
        Y = tsne.fit_transform(X)

        for touch in y_items:
            print(touch)
            c_plot_bool = y == touch
            ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], label= "touch : {}".format(touch))
        ax.legend()
        plt.savefig("./t_sne/touch_komavec_tsne_" + str(n) + "_" + str(p) + ".png")
        plt.clf()
