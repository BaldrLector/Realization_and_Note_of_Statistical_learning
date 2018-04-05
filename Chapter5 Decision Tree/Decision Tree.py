from collections import Counter

import numpy as np
import math

# data : 年龄 有工作 有自己的房子 信贷情况 类别
X = [
    ['青年', '否', '否', '一般', '否'],
    ['青年', '否', '否', '好', '否'],
    ['青年', '是', '否', '好', '是'],
    ['青年', '是', '是', '一般', '是'],
    ['青年', '否', '否', '一般', '否'],

    ['中年', '否', '否', '一般', '否'],
    ['中年', '否', '否', '好', '否'],
    ['中年', '是', '是', '好', '是'],
    ['中年', '否', '是', '非常好', '是'],
    ['中年', '否', '是', '非常好', '是'],

    ['老年', '否', '是', '非常好', '是'],
    ['老年', '否', '是', '好', '是'],
    ['老年', '是', '否', '好', '是'],
    ['老年', '是', '否', '非常好', '是'],
    ['老年', '否', '否', '一般', '否'],
]

X = np.array(X)
Y = X[:, -1]
N = X.shape[0]
print('X:\t', X)
print('Y:\t', Y)

def info_gain(dataset, label, feature):
    # here fearture is index of dataset dimension
    c = Counter(dataset[:, feature])
    n = len(dataset)
    hd = 0.0

    # compute entropy H(D)
    for i in Counter(label):
        p = Counter(label)[i] / n
        hd -= p * math.log2(p)

    # compute H(D|A)
    hda = 0.0
    for i in Counter(dataset[:, feature]):
        lebel_i = label[dataset[:, feature] == i]
        hik = 0.0
        for k in Counter(lebel_i):
            pik = Counter(lebel_i)[k] / len(lebel_i)
            hik += pik * math.log2(pik)
        hda -= len(lebel_i) / len(label) * hik
    return hd - hda


def info_gain_ratio(dataset, label, feature):
    gda = info_gain(dataset, label, feature)

    # Compute H_A(D)
    had = 0.0
    for i in Counter(dataset[:, feature]):
        p = Counter(dataset[:, feature])[i] / len(dataset)
        had -= p * math.log2(p)

    return gda / had
