"""
============================
# -*- coding: utf-8 -*-
# Time    : 2022/10/18 15:35
# Author  : Qisx
# FileName: LogisticRegression4Diabetes.py
# Software: PyCharm
===========================
"""
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn import Module, Linear, Sigmoid
from torch.nn import BCELoss
from torch.optim import SGD


# prepare dataset
xy = np.loadtxt('Datasets/DiabetesCSV/diabetes.csv', delimiter=',', dtype=np.float32)
x_train = torch.from_numpy(xy[:, :-1])
y_train = torch.from_numpy(xy[:, [-1]])


# design model
class LogisticRegression(Module):

    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.linear1 = Linear(8, 4)  # 输入数据是八维的
        self.linear2 = Linear(4, 2)
        self.linear3 = Linear(2, 1)
        self.sigmoid = Sigmoid()  # 将结果投影到概率层

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = LogisticRegression()
# construct loss and optimizer
criterion = BCELoss(reduction='mean')
optimizer = SGD(model.parameters(), lr=0.01)

loss_list = []
epoch_list = []
# training
for epoch in range(1000000):
    y_predict = model(x_train)
    loss = criterion(y_predict, y_train)

    loss_list.append(loss.item())
    epoch_list.append(epoch)

    optimizer.zero_grad()  # 下次运算之前清除梯度信息
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    if epoch % 1000 == 999:
        y_pred_label = torch.where(y_predict >= 0.5, torch.tensor([1.0]), torch.tensor([0.0]))

        acc = torch.eq(y_pred_label, y_train).sum().item() / y_train.size(0)
        print("loss = ", loss.item(), "acc = ", acc)

plt.figure()
plt.plot(epoch_list, loss_list)
plt.xlabel("Epoch")
plt.ylabel("COST")
plt.show()
