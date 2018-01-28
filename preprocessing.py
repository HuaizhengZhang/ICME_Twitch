#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/1/18 2:20 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : preprocessing.py

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import pickle
from config import cfg
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing


def read_data(path):
    dataset = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)
    x = dataset[:, 0:-1]
    y = dataset[:, -1]
    min_max = preprocessing.MinMaxScaler()
    x_min_max = min_max.fit_transform(x)
    # print(x.shape, y.shape)

    return x_min_max, y


def vis(x, y):
    d = {'bitrate': x[:, 2], 'bandwidth': y}
    df = pd.DataFrame(data=d)
    print(df)
    sns.jointplot(x='bitrate', y='bandwidth', data=df)
    plt.show()


def seperate_data(path):
    if len(os.listdir(cfg.TRAINING_SEGMENT_IDX)) == 0:
        dataset = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=float)
        diff_segment = np.diff(dataset[:, -2])
        split_idx = np.where(diff_segment > 0)[0]
        split_idx += 1
        split = np.split(dataset, split_idx)

        for i, value in enumerate(split[0:10]):
            with open(cfg.TRAINING_SEGMENT_IDX + str(i) + '.pkl', 'wb') as f:
                pickle.dump(value, f)
            print(i)
            print (len(value))
            # temp = np.where(np.diff(value[:, -1]) > 0)[0]
            # if len(temp) > 0:
            #     print(i)
    else:
        print('Finished')

