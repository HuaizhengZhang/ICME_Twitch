#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 22/1/18 2:26 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : dataloader.py

import torch
import numpy as np
from torch.utils.data import Dataset

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
        

