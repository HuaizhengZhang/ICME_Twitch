#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/1/18 2:26 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : dataloader.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import pickle

class abrDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        arr_time = self.x[idx, 0]
        bytes = self.x[idx, 1]
        bitrate = self.x[idx, 2]
        segment_idx = self.x[idx, 3]

        bandwidth = self.y[idx]

        sample = {'arr_time': arr_time, 'bytes': bytes, 'bitrate': bitrate,
                  'segment_idx': segment_idx, 'bandwidth': bandwidth}

        return sample

class segDataset(Dataset):
    def __init__(self, path):
        # This part has no order!
        self.segments = [os.path.join(path, i) for i in os.listdir(path)]

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        with open(self.segments[idx], 'rb') as f:
            data = pickle.load(f)[:, 0:-1]
            x = data[:, 0:-1]
            y = data[:, -1]



