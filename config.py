#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/1/18 2:21 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : config.py

from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C.TRAINING_DATA_PATH = 'data/training.csv'

__C.TRAINING_RATIO = 0.9

__C.BATCH_SIZE = 1

__C.EPOCHS = 1000

__C.MODEL_SAVE = 'model/'

__C.TRAINING_SEGMENT_PATH = 'data/train_seg/'

__C.TRAINING_SEGMENT_IDX = 'data/train_seg/segment_index_'

__C.LSTM_INPUT_SIZE = 28
__C.LSTM_HIDDEN_SIZE = 28
__C.LSTM_NUMBER_LAYERS = 2