#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 21/1/18 2:59 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : train_NN.py

from __future__ import print_function

import os
import numpy as np
import torch
from datetime import datetime
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
from config import cfg
from DL_based_method.nets import *
from DL_based_method.dataloader import *
from preprocessing import read_data

def main():
    torch.cuda.set_device(1)
    model = baseNN()
    model = model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x, y = read_data(cfg.TRAINING_DATA_PATH)

    train_size = int(cfg.TRAINING_RATIO * len(x))

    x_train = x[:train_size]
    y_train = y[:train_size]

    x_val = x[train_size:]
    y_val = y[train_size:]

    train_data = abrDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=False)

    for epoch in range(cfg.EPOCHS):
        for batch_idx, sample_batched in enumerate(train_loader):
            pid = os.getpid()
            x_1 = Variable(sample_batched['arr_time'].type(torch.FloatTensor).cuda())
            x_2 = Variable(sample_batched['bytes'].type(torch.FloatTensor).cuda())
            x_3 = Variable(sample_batched['bitrate'].type(torch.FloatTensor).cuda())
            x_4 = Variable(sample_batched['segment_idx'].type(torch.FloatTensor).cuda())

            target = Variable(sample_batched['bandwidth'].type(torch.FloatTensor).cuda())

            optimizer.zero_grad()
            prediction = model(x_1, x_2, x_3, x_4)

            # print (target)
            loss = F.mse_loss(prediction, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, epoch, batch_idx * cfg.BATCH_SIZE, len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), loss.data[0]))
    torch.save(model.state_dict(), cfg.MODEL_SAVE + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.pt')

    test_data = abrDataset(x_val, y_val)
    test_loader = data_utils.DataLoader(test_data, batch_size=cfg.BATCH_SIZE, shuffle=False)
    model.eval()
    test_loss = 0
    for sample_batched in test_loader:
        x_1 = Variable(sample_batched['arr_time'].type(torch.FloatTensor).cuda())
        x_2 = Variable(sample_batched['bytes'].type(torch.FloatTensor).cuda())
        x_3 = Variable(sample_batched['bitrate'].type(torch.FloatTensor).cuda())
        x_4 = Variable(sample_batched['segment_idx'].type(torch.FloatTensor).cuda())

        target = Variable(sample_batched['bandwidth'].type(torch.FloatTensor).cuda())

        output = model(x_1, x_2, x_3, x_4)
        test_loss += F.mse_loss(output, target, size_average=False).data[0]  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    main()