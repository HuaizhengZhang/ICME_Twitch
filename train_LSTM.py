#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 26/1/18 1:44 PM
# @Author  : Huaizheng Zhang
# @Site    : zhanghuaizheng.info
# @File    : train_LSTM.py

import torch
import torch.optim as optim
import torch.utils.data as data_utils
from config import cfg
from DL_based_method.nets import *
from DL_based_method.dataloader import *


def main():
    torch.cuda.set_device(1)
    model = baseLSTM(4, 512, 1, cfg.BATCH_SIZE)
    model.cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_data = segDataset(cfg.TRAINING_SEGMENT_PATH)

    def my_collate(batch):
        data = torch.cat([torch.from_numpy(item['segment']) for item in batch], 0)
        label = np.array([item['bandwidth'] for item in batch])
        label = torch.LongTensor(label)
        return [data, label]
    train_loader = data_utils.DataLoader(train_data, batch_size=cfg.BATCH_SIZE, shuffle=False)


    for epoch in range(cfg.EPOCHS):
        for batch_idx, sample_batched in enumerate(train_loader):
            pid = os.getpid()
            # print (sample_batched['segment'])
            seg = Variable(sample_batched['segment']).cuda()
            label = Variable(sample_batched['bandwidth'].type(torch.FloatTensor).cuda())

            optimizer.zero_grad()

            model.hidden = model.init_hidden()

            prediction = model(seg)
            loss = F.mse_loss(prediction, label)
            loss.backward(retain_graph=True)
            optimizer.step()

            if batch_idx % 3 == 0:
                print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    pid, epoch, batch_idx * cfg.BATCH_SIZE, len(train_loader.dataset),
                                100. * batch_idx / len(train_loader), loss.data[0]))
        torch.save(model.state_dict(), cfg.MODEL_SAVE + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '.pt')

if __name__ == '__main__':
    main()