# coding:utf-8

import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class LR(object):

    def fit(self, x, y, iterations=1000):
        self.w = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

    def predict(self, x):
        return x.dot(self.w)


x, y = make_moons(250, noise=0.3)

# print(x.shape)  # (250, 2)
# print(y.shape)  # (250,)

lr = LR()
lr.fit(x, y.reshape(-1, 1))

plt.figure()
for i in range(x.shape[0]):
    col = {0: 'r', 1: 'b'}
    plt.plot(x[i, 0], x[i, 1], col[y[i]] + 'o')

plt.ylim([-1.5, 1.5])
xpts = np.linspace(-1.5, 2.5)
ypts = (lr.w[0] * xpts) / (-lr.w[1])

plt.plot(xpts, ypts, 'g-', lw=2)
plt.show()
