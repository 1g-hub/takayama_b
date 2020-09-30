# coding: utf-8
path = "../models/bert/hottoSNS-bert-pytorch/m.txt"

with open(path) as f:
    s = f.read()
    print(type(s))
    print(s)

print(type(f))
# <class '_io.TextIOWrapper'>

f.close()