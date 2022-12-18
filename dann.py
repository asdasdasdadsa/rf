from random import shuffle

import numpy as np

class dann(object):

    def normaniz(self, x):
        Xt = np.array(x).T
        for i in range(len(Xt)):
            x_min = np.min(Xt[i])
            x_max = np.max(Xt[i])
            if (x_max - x_min) != 0:
                Xt[i] = (Xt[i] - x_min) / (x_max - x_min)
        return Xt.T

    def standortiz(self, x):
        Xt = np.array(x).T
        for i in range(len(Xt)):
            srednee = np.average(Xt[i])
            disp = np.var(Xt[i])
            if disp != 0:
                Xt[i] = (Xt[i] - srednee) / disp
        return Xt.T

    def one_hot_encoding(self, t):
        H = []
        for i in range(len(t)):
            h = [0 for j in range(10)]
            h[t[i]] = 1
            H.append(h)
        return np.array(H)

    def peremeshat_razd(self, x, t, percent=None):#не придумал как перемешать 2 массива одинаково по дургому
        if percent is None:
            percent = [0.8, 0.1, 0.1]
        N = len(x)
        k = [[i, x[i], t[i]] for i in range(N)]
        shuffle(k)
        X_train, X_val, X_test, t_train, t_val, t_test = [], [], [], [], [], []
        for i in range(N):
            if i < int(N * percent[0]):
                X_train.append(k[i][1])
                t_train.append(k[i][2])
            if int(N * percent[0]) <= i < int(N * (percent[1] + percent[0])):
                X_val.append(k[i][1])
                t_val.append(k[i][2])
            if i >= int(N * (percent[1] + percent[0])):
                X_test.append(k[i][1])
                t_test.append(k[i][2])
        return np.array(X_train), np.array(X_val), np.array(X_test), \
               np.array(t_train), np.array(t_val), np.array(t_test)
