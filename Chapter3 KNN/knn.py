# coding:utf-8
import numpy as np


class knn(object):

    def __init__(self):
        pass

    def train(self, x, y):
        # X.shape: N,D ;  N = num_train
        # y.shape: N,
        self.X_train = x
        self.Y_train = y

    def predict(self, x, k=10):
        # x.shape num_test,D ; num_test=x.shape[0]
        # return N,
        num_test = x.shape[0]
        num_train = self.X_train.shae[0]
        distance = self.get_distance(x)  # distance.shape : num_test,num_train

        y_predict = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.Y_train[np.argsort(distance[i])[0:k]]
            y_predict[i] = np.bincount(closest_y).argmax()

    def get_distance(self, x):
        num_test = x.shape[0]
        num_train = self.X_train.shae[0]
        dis = np.zeros((num_test, num_train))
        dis += np.sum(x ** 2, axis=1).reshape(num_test, 1)
        dis += np.sum((self.X_train ** 2).sum(axis=1)).reshape(1, num_train)
        dis -= 2 * np.dot(x, self.X_train.T)
        return dis
