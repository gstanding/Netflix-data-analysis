# author: viaeou
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import numpy.matlib

rating = pd.read_csv('data/csv/ml-latest-small/ratings.csv')
rating_matrix = pd.pivot_table(rating[['userId', 'movieId', 'rating']], index='userId', columns='movieId', values='rating')
normal_matrix = np.matlib.randn(rating_matrix.shape)
print(normal_matrix.shape, normal_matrix[:5, :])
train_trans = (~rating_matrix.isna().values) * np.array(np.abs(normal_matrix) <= 1)
test_trans = (~rating_matrix.isna().values) * np.array(np.abs(normal_matrix) > 1)
rating_matrix = rating_matrix.fillna(0)

r_matrix = rating_matrix.values


class Baseline(object):

    def __init__(self, rui, train_trans, test_trans, learning_rate=0.01, n_iter=400, lambda_3 = 0):
        self.learning = learning_rate
        self.n_iter = n_iter
        self.lambda_3 = lambda_3
        self.rui = rui
        self.mu = np.sum(rui)/100004
        #print('mu: ', self.mu)
        self.bu = np.zeros(shape=(rui.shape[0], 1))
        self.bi = np.zeros(shape=(1, rui.shape[1]))
        self.train_trans = train_trans
        self.test_trans = test_trans

    def compute_cost(self):
        cost = np.sum(np.power((self.rui - self.mu - self.bu - self.bi) * self.train_trans, 2)) + \
               self.lambda_3 * np.sum((np.power(self.bu, 2)) + np.sum(np.power(self.bi, 2)))
        # print(self.rui.shape, self.bu.shape, self.bi.shape)
        print(cost)
        return cost

    def fit(self):
        costs = []
        for i in range(self.n_iter):
            temp = (self.rui - self.mu - self.bu - self.bi)*self.train_trans
            self.bu += self.learning * (np.sum(temp, axis=1, keepdims=True)/1000 - \
                       self.lambda_3 * self.bu)
            self.bi += self.learning * (np.sum(temp, axis=0, keepdims=True)/1000 - \
                       self.lambda_3 * self.bi)
            print('bu', self.bu[:5, 0])
            print('bi', self.bi[0, :5])
            cost = self.compute_cost()
            costs.append(cost)
            if i % 100 == 0:
                print('cost: ', cost)

    def predict(self):
        bui = (self.mu + self.bu + self.bi) * self.test_trans
        rmse = np.sqrt(np.sum(np.power((bui - self.rui * self.test_trans), 2))/(100004*(1-0.68268949)))

        print(rmse)


baseline = Baseline(r_matrix, train_trans, test_trans)
cost = baseline.compute_cost()
baseline.fit()
baseline.predict()
