# author: viaeou
# author: viaeou
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import numpy.matlib


class BaselineTime(object):

    def __init__(self, ratings, learning_rate=.005, n_epoches=20, reg=.02, beta=0.4, n_bins=30):
        self.lr = learning_rate
        self.n_epoches = n_epoches
        self.reg = reg
        self.mu = np.mean(ratings, axis=0)[2]
        self.bu = np.zeros(shape=len(set(ratings[:, 0])) + 1)
        self.bi = np.zeros(shape=int(ratings[:, 1].max() + 1))
        self.alpha_u = np.zeros((len(set(ratings[:, 0])) + 1, 1))
        self.bit = np.zeros(shape=(int(ratings[:, 1].max() + 1), n_bins))
        self.but = np.zeros(shape=(int(ratings[:, 0].max() + 1),
                                   int(ratings[:, 3].max() - ratings[:, 3].min() + 1)))
        self.early_day = int(ratings[:, 3].min())
        self.beta = beta
        self.dev = np.zeros(shape=(int(ratings[:, 3].max() - ratings[:, 3].min() + 1), 1))

    def compute_cost(self, r_train):
        cost = 0
        for uid, iid, rui, day, bin, tu_mean in r_train:
            uid, iid, day, bin, tu_mean = int(uid), int(iid), int(day), int(bin), int(tu_mean)
            self.dev[day - self.early_day] = np.sign(day-tu_mean)*abs(day-tu_mean) ** self.beta
            cost += (rui - self.mu - self.bu[uid] - self.alpha_u[uid] * self.dev[day - self.early_day] - self.but[uid, day - self.early_day]
                     - self.bi[iid] - self.bit[iid, bin]) ** 2 + \
                    self.reg * (self.bu[uid] ** 2 + self.bi[iid] ** 2 + self.alpha_u[uid] ** 2 +
                        self.but[uid, day - self.early_day] ** 2 + self.bit[iid, bin] ** 2)
        return cost

    def fit(self, r_train):
        for epoch in range(self.n_epoches):
            cost = self.compute_cost(r_train)
            cnt = 0
            for uid, iid, rui, day, bin, tu_mean in r_train:
                cnt += 1
                if cnt == 100:
                    break
                uid, iid, day, bin, tu_mean = int(uid), int(iid), int(day), int(bin), int(tu_mean)
                err = rui - self.mu - self.bu[uid] - self.alpha_u[uid] * self.dev[day - self.early_day] - \
                      self.but[uid, day - self.early_day] - self.bi[iid] - self.bit[iid, bin]
                print('error:', err)
                self.bu[uid] += self.lr * (err - self.reg * self.bu[uid])
                self.alpha_u[uid] += self.lr * (err * self.dev[day - self.early_day] - self.reg * self.alpha_u[uid])
                self.bi[iid] += self.lr * (err - self.reg * self.bi[iid])
                self.but[uid, day - self.early_day] += self.lr * (err - self.reg *
                                                                  self.but[uid, day - self.early_day])
                self.bit[iid, bin] += self.lr * (err - self.reg * self.bit[iid, bin])
            print('error:', err)
            print(self.bu[:3], self.alpha_u[:3], self.bi[:3], self.but[:3, :3], self.bit[:3, :3])
            print('Epoch %d: cost: %.6f'%(epoch, cost))

    def predict(self, r_test):
        square_error = 0
        for uid, iid, rui in r_test.astype(int):
            bui = self.mu + self.bu[uid] + self.bi[iid]
            square_error += (bui - rui) ** 2
        rmse = np.sqrt(square_error / r_test.shape[0])
        print(rmse)


rating = pd.read_csv('data/csv/ml-latest-small/ratings.csv')
rating['days'] = (rating['timestamp'] / 3600 / 24).astype(int)
tu_means = rating.groupby('userId')['days'].mean().astype(int)
tu_means = pd.DataFrame(tu_means)
tu_means.columns = ['tu_means']
# cut_bins


def cut(x, n_bins = 30):
    return pd.cut(x, bins=n_bins, labels=range(n_bins))


rating['bins'] = rating.groupby('movieId')['days'].apply(cut)
rating = pd.merge(rating, tu_means, on='userId')
# print(rating.head())
ratings = rating[['userId', 'movieId', 'rating', 'days', 'bins', 'tu_means']]
# print(ratings.columns)
ratings = ratings.values

baseline = BaselineTime(ratings)
r_train, r_test = train_test_split(ratings, shuffle=True, train_size=.75, test_size=.25, random_state=14)
# baseline.compute_cost(r_train)
baseline.fit(r_train)
# baseline.predict(r_test)
