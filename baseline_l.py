# coding:utf-8

from surprise import AlgoBase
from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import PredictionImpossible
from surprise.model_selection import cross_validate
from surprise import accuracy
import numpy as np
import pandas as pd


class BaselineWithTime(AlgoBase):
    def __init__(self, n_epochs=20, biased=True,
                 lr_all=.005, reg_all=.02,lr_bu=None,
                 lr_bi=None, reg_bu=None, reg_bi=None,
                 verbose=True):
        self.n_epochs = n_epochs
        self.biased = biased
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.verbose = verbose
        AlgoBase.__init__(self)

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        self.sgd(trainset)
        return self

    def sgd(self, trainset):
        global_mean = self.trainset.global_mean
        lr_bu = self.lr_bu
        lr_bi = self.lr_bi
        reg_bu = self.reg_bu
        reg_bi = self.reg_bi
        bu = np.zeros(trainset.n_users, np.double)
        bi = np.zeros(trainset.n_items, np.double)
        if not self.biased:
            global_mean = 0
        for current_epoch in range(self.n_epochs):
            if self.verbose:
                print("Processing epoch {}".format(current_epoch))
            for u, i, r in trainset.all_ratings():
                err = r - (global_mean + bu[u] + bi[i])
                # update biases
                if self.biased:
                    bu[u] += lr_bu * (err - reg_bu * bu[u])
                    bi[i] += lr_bi * (err - reg_bi * bi[i])
        self.bu = bu
        self.bi = bi

    def estimate(self, u, i):
        known_user = self.trainset.knows_user(u)
        known_item = self.trainset.knows_item(i)
        if self.biased:
            est = self.trainset.global_mean
            if known_user:
                est += self.bu[u]
            if known_item:
                est += self.bi[i]
        else:
            raise PredictionImpossible('User and item are unkown.')

        return est


# Retrieve the trainset.
# trainset = data.build_full_trainset()

# Use the famous SVD algorithm.
# algo = SVD()

# Run 5-fold cross-validation and print results.
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the algorithm on the trainset, and predict ratings for the testset
# predictions = algo.fit(trainset).test(testset)

# Then compute RMSE
# accuracy.rmse(predictions)

# Build an algorithm, and train it.
# algo = KNNBasic()
# algo.fit(trainset)

# uid = str(196)
# iid = str(302)
# algo.predict(uid, iid, r_ui=4, verbose=True)

# path to dataset file
# file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')

# As we're loading a custom dataset, we need to define a reader. In the
# movielens-100k dataset, each line has the following format:
# 'user item rating timestamp', separated by '\t' characters.
# reader = Reader(line_format='user item rating timestamp', sep='\t')

# data = Dataset.load_from_file(file_path, reader=reader)

# sample random trainset and testset
# test set is made of 25% of the ratings.
rating = pd.read_csv('data/csv/ml-latest-small/ratings.csv')
reader = Reader()

data = Dataset.load_from_df(rating[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=.25)

algo = BaselineWithTime()

cross_validate(algo, data, verbose=True)

predictions = algo.fit(trainset).test(testset)

accuracy.rmse(predictions)
