# author: viaeou
import time
import tensorflow as tf

import pandas as pd


class DataProcess(object):

    def __init__(self, path):
        self.path = path

    def load_data(self):
        with open(self.path, 'r') as f:
            # for i in range(2):
            #     print(f.readline())
            counts_per_movie = {}
            counts_per_user = {}
            users_list = []
            count = 0
            # start = time.time()
            while True:
                count += 1
                if count % 100000 == 0:
                    print(count)
                line = f.readline()
                if not line:
                    break
                elif ':' in line:
                    flag = int(line.split(':')[0])
                    counts_per_movie[flag] = 0
                else:
                    user_id = int(line.split(',')[0])
                    if user_id not in users_list:
                        users_list.append(user_id)
                        counts_per_user[user_id] = 1
                    counts_per_user[user_id] += 1
                    counts_per_movie[flag] += 1
            # print(counts_per_movie, time.time() - start)
            counts_per_movie = pd.DataFrame(counts_per_movie, index=['users_counts'])
            counts_per_user = pd.DataFrame(counts_per_user, index=['movies_counts'])
            # print(counts_per_movie)
            return counts_per_movie, counts_per_user


if __name__ == '__main__':
    start = time.time()
    print(start)
    part_1 = DataProcess('F:\deep learning\data\\Netflix\combined_data_1.txt')
    part_2 = DataProcess('F:\deep learning\data\\Netflix\combined_data_2.txt')
    part_3 = DataProcess('F:\deep learning\data\\Netflix\combined_data_3.txt')
    part_4 = DataProcess('F:\deep learning\data\\Netflix\combined_data_4.txt')
    print(time.time()-start)
    data_10, data_11 = part_1.load_data()
    data_20, data_21 = part_2.load_data()
    data_30, data_31 = part_3.load_data()
    data_40, data_41 = part_4.load_data()
    # print(data_1.shape, data_2.shape, data_3.shape, data_4.shape)
    # print(data_1, data_2, data_3, data_4)
    counts_per_movie = pd.concat([data_10, data_20, data_30, data_40], axis=1).T
    counts_per_user = pd.concat([data_11, data_21, data_31, data_41], axis=1).T
    counts_per_movie.to_csv('./csv/counts_per_movie.csv')
    counts_per_user.to_csv('./csv/counts_per_user.csv')
    print(time.time() - start)

