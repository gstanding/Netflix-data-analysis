# author: viaeou
# author: viaeou
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

rating = pd.read_csv('data/csv/ml-latest-small/ratings.csv')
#rating = rating[['userId', 'movieId', 'rating']]
rating['days'] = (rating['timestamp'] / 3600 / 24).astype(int)
tu_mean = rating.groupby('userId')['days'].mean().astype(int)

def cut(x):
    return pd.cut(x, bins=30, labels=range(30))


rating['bins'] = rating.groupby('movieId')['days'].apply(cut)
ratings = rating.values
print(rating.head(100))
#
# class BaselineTime(object):
#
#     def __init__(self, ratings, learning_rate=.005, n_epoches=20, reg=.02, n_bins = 30):
#         self.lr = learning_rate
#         self.n_epoches = n_epoches
#         self.reg = reg
#         self.mu = np.mean(ratings, axis=0)[2]
#         self.bu = np.zeros(shape=len(set(ratings[:, 0])) + 1)
#         self.bi = np.zeros(shape=int(ratings[:, 1].max() + 1))
#         #time bins
#         self.n_bins = n_bins
#
#     def compute_cost(self, r_train):
#         cost = 0
#         for uid, iid, rui in r_train.astype(int):
#             cost += (rui - self.mu - self.bu[uid] - self.bi[iid]) ** 2 + \
#                     self.reg * (self.bu[uid] ** 2 + self.bi[iid] ** 2)
#         return cost
#
#     def fit(self, r_train):
#         for epoch in range(self.n_epoches):
#             for uid, iid, rui in r_train.astype(int):
#                 err = rui - (self.mu + self.bu[uid] + self.bi[iid])
#                 self.bu[uid] += self.lr * (err - self.reg * self.bu[uid])
#                 self.bi[iid] += self.lr * (err - self.reg * self.bi[iid])
#             cost = self.compute_cost(r_train)
#             print('Epoch %d: cost: %.6f'%(epoch, cost))
#
#     def predict(self, r_test):
#         square_error = 0
#         for uid, iid, rui in r_test.astype(int):
#             bui = self.mu + self.bu[uid] + self.bi[iid]
#             square_error += (bui - rui) ** 2
#         rmse = np.sqrt(square_error / r_test.shape[0])
#         print(rmse)
#
#
# baseline = BaselineTime(ratings)
# r_train, r_test = train_test_split(ratings, shuffle=True, train_size=.75, test_size=.25)
# baseline.fit(r_train)
# baseline.predict(r_test)
