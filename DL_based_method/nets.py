#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/1/18 4:52 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : nets.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class baseNN(nn.Module):
    def __init__(self):
        super(baseNN, self).__init__()
        self.input_layer = nn.Linear(4, 64)
        self.fc1 = nn.Linear(64, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 1)

    def forward(self, x_1, x_2, x_3, x_4):
        # print x_1.view(-1, 1).size(), x_2.view(-1, 1).size(), x_3.view(-1, 1).size(), x_4.view(-1, 1).size()
        h = torch.cat((x_1.view(-1, 1), x_2.view(-1, 1), x_3.view(-1, 1), x_4.view(-1, 1)), 1)
        # print h
        h = self.input_layer(h)
        h = F.relu(h)

        h = self.fc1(h)
        h = self.bn1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.25, training=self.training)

        h = self.fc2(h)
        h = self.bn2(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.25, training=self.training)

        h = self.fc3(h)

        return h

class baseLSTM(nn.Model):
    def __init__(self, input_size, hidden_size, num_layers):
        super(baseLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, 1)
        self.hidden_state = self.init_hidden()

    def init_hidden(self, x):
        return (Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda(),
                Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda())

    def forward(self, x):
        lstm_out, self.hidden_state = self.lstm(x, self.hidden_state)
        out = self.fc1(lstm_out[: -1, :])
        return out


