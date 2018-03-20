# coding:utf-8
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


class LR(object):

    def __init__(self, D):
        self.w = np.random.randn(D, 1)
        self.b = 0

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, x):
        return self.w.T.dot(x) + self.b

    def fit(self, x, y, it=10000, a=0.1):
        # N,D   x.shape
        # 1,N   y.shape
        # 1,D   w.shape
        pass
        N, D = x.shape

        for i in range(it):
            h = self.sigmoid(self.w.T.dot(x) + self.b)
            # print(h.shape) (1, 250)
            dw = np.sum((h - y) * x, axis=1) / N * a
            # print(dw.shape)  (2,)
            db = np.sum(h - y, keepdims=False) / N * a
            self.w -= dw.reshape(2, 1)
            self.b -= db

    def predict(self, x):
        return self.w.T.dot(x) + self.b


if __name__ == '__main__':
    x, y = make_moons(250, noise=0.25)

    # print(x.shape)  # (250, 2)
    # print(y.shape)  # (250,)

    col = {0: 'r', 1: 'b'}
    lr = LR(2)
    lr.fit(x.T, y.reshape(1, -1))

    #show result
    plt.figure()
    for i in range(x.shape[0]):
        col = {0: 'r', 1: 'b'}
        plt.plot(x[i, 0], x[i, 1], col[y[i]] + 'o')
    plt.ylim([-1.5, 1.5])

    xpts = np.linspace(-1.5, 2.5)
    ypts = (lr.w[0] * xpts) / (-lr.w[1])

    plt.plot(xpts, ypts, 'g-', lw=2)
    plt.show()
