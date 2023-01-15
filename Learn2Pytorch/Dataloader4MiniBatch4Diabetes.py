"""
============================
# -*- coding: utf-8 -*-
# Time    : 2022/10/20 16:52
# Author  : Qisx
# FileName: Dataloader4MiniBatch4Diabetes.py
# Software: PyCharm
===========================
"""
import numpy as np
import torch
from matplotlib import pyplot as plt

from torch.nn import Module, Linear, Sigmoid, BCELoss,NLLLoss
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader


# prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_train = torch.from_numpy(xy[:, :-1])
        self.y_train = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, item):
        return self.x_train[item], self.y_train[item]

    def __len__(self):
        return self.len


filepath = 'Datasets/DiabetesCSV/diabetes.csv'
dataset = DiabetesDataset(filepath)
train_load = DataLoader(dataset=dataset,
                        batch_size=32,
                        shuffle=True,
                        )


# design model
class DiabetesModel(Module):
    def __init__(self):
        super(DiabetesModel, self).__init__()
        self.linear1 = Linear(8, 4)
        self.linear2 = Linear(4, 1)
        self.sigmoid = Sigmoid()

    def forward(self, inputs):
        x = self.sigmoid(self.linear1(inputs))
        x = self.sigmoid(self.linear2(x))
        return x


model = DiabetesModel()

# construct loss and optimizer
criterion = BCELoss(reduction='mean')
optimizer = SGD(model.parameters(), lr=0.01)

# training
for epoch in range(5000):
    for i, data in enumerate(train_load, 0):
        # step1: prepare data
        inputs, labels = data
        # step2: forward
        y_predict = model(inputs)
        loss = criterion(y_predict, labels)

        # step3：backward
        optimizer.zero_grad()
        loss.backward()
        # step4 update
        optimizer.step()
    print(epoch, i, loss.item())

