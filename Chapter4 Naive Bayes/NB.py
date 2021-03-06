import math
import time
import numpy as np
from collections import Counter
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def calcProbDensity(meanLabel, stdLabel, test_X):  # 计算某样本的条件概率密度P(xi|c)
    numAttributes = len(test_X)
    MultiProbDensity = 1.0
    for i in range(numAttributes):  # 对样本test_X每个属性的概率密度函数连乘
        MultiProbDensity = np.exp(-np.square(test_X[i] - meanLabel[i]) / (2.0 * np.square(stdLabel[i]))) / (
                np.sqrt(2.0 * math.pi) * stdLabel[i]) * MultiProbDensity
    return MultiProbDensity


def calcPriorProb(Y_train):  # 计算每一类的先验概率P(c)
    i, j = 0, 0
    global labelValue, classNum
    numSamples = Y_train.shape[0]
    labelValue = np.zeros((numSamples, 1))  # 前i行用来保存标签值
    Y_train_counter = sum(Y_train.tolist(), [])  # 将Y_train转化为可哈希的数据结构
    cnt = Counter(Y_train_counter)  # 计算标签值的类别个数及各类样例的个数
    for key in cnt:
        labelValue[i] = key
        i += 1
    classNum = i  # 类别总数
    Pc = np.zeros((classNum, 1))  # 不同类的先验概率
    eachLabelNum = np.zeros((classNum, 1))  # 每类样例数
    for key in cnt:
        Pc[j] = cnt[key] / numSamples
        eachLabelNum[j] = cnt[key]
        j += 1
    return labelValue, eachLabelNum, classNum, Pc


def trainBayes(X_train, Y_train):
    startTime = time.time()
    numTrainSamples, numAttributes = X_train.shape
    labelValue, eachLabelNum, classNum, Pc = calcPriorProb(Y_train)
    meanlabelX, stdlabelX = [], []  # 存放每一类样本在所有属性上取值的均值和方差
    for i in range(classNum):
        k = 0
        labelXMatrix = np.zeros((int(eachLabelNum[i]), numAttributes))  # 取出某一类所有样本组成新矩阵
        for j in range(numTrainSamples):
            if Y_train[j] == labelValue[i]:
                labelXMatrix[k] = X_train[j, :]
                k += 1
        meanlabelX.append(np.mean(labelXMatrix, axis=0).tolist())  # 求该矩阵的列均值与无偏标准差，append至所有类
        stdlabelX.append(np.std(labelXMatrix, ddof=1, axis=0).tolist())
    meanlabelX = np.array(meanlabelX).reshape(classNum, numAttributes)
    stdlabelX = np.array(stdlabelX).reshape(classNum, numAttributes)
    print('---Train completed.Took %f s.' % ((time.time() - startTime)))
    return meanlabelX, stdlabelX, Pc


def predict(X_test, Y_test, meanlabelX, stdlabelX, Pc):
    numTestSamples = X_test.shape[0]
    matchCount = 0
    for m in range(X_test.shape[0]):
        x_test = X_test[m, :]  # 轮流取测试样本
        pred = np.zeros((classNum, 1))  # 对不同类的概率
        for i in range(classNum):
            pred[i] = calcProbDensity(meanlabelX[i, :], stdlabelX[i, :], x_test) * Pc[i]  # 计算属于各类的概率
        predict = labelValue[np.argmax(pred)]  # 取出标签
        if predict == Y_test[m]:
            matchCount += 1
    accuracy = float(matchCount / numTestSamples)
    return accuracy


if __name__ == '__main__':
    print('Step 1.Loading data...')
    # 数据集下载http://download.csdn.net/download/chai_zheng/10009919
    data = np.loadtxt("./Wine.txt", delimiter=',')  # 载入葡萄酒数据集
    print('---Loading completed.')
    x = data[:, 1:14]
    y = data[:, 0].reshape(178, 1)
    print('Step 2.Splitting and preprocessing data...')
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.4)  # 拆分数据集
    scaler = preprocessing.StandardScaler().fit(X_train)  # 数据标准化
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print('---Splittinging completed.\n---Number of training samples:%d\n---Number of testing samples:%d' \
          % (X_train.shape[0], X_test.shape[0]))
    print('Step 3.Training...')
    meanlabelX, stdlabelX, Pc = trainBayes(X_train, Y_train)
    print('Step 4.Testing...')
    accuracy = predict(X_test, Y_test, meanlabelX, stdlabelX, Pc)
    print('---Testing completed.Accuracy:%.3f%%' % (accuracy * 100))
