#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
# @Time    : 18-8-22 上午11:01
# @Author  : viaeou
# @Site    : 
# @File    : example.py
# @Software: PyCharm


import os
from surprise import SVD, SVDpp
from surprise import Dataset
from surprise.model_selection import cross_validate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the movielens-100k dataset (download it if needed),
data = Dataset.load_builtin('ml-100k')

# We'll use the famous SVD algorithm.
#algo = SVD()
algo = SVDpp()

# Run 5-fold cross-validation and print results
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)