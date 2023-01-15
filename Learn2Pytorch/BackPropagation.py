"""
============================
# -*- coding: utf-8 -*-
# Time    : 2022/10/12 17:22
# Author  : Qisx
# FileName: BackPropagation.py
# Software: PyCharm
===========================
"""
import torch


def forward(x):
    """
    前推模型
    :param x:训练集x值
    :param w: 模型权值，Tensor类型
    :return:
    """
    return x * w


def loss(x, y):
    """
    计算训练值与预测结果的误差
    :param y: 训练集实际值
    :param y: 训练集预测值
    :param w: 模型权值，Tensor类型
    :return: 误差的平方和
    """
    y_predict = forward(x)
    return (y_predict - y) ** 2


w = torch.tensor([1.0])
w.requires_grad = True  # 需计算梯度,默认为False


def main():
    x_train = [1.0, 2.0, 3.0]
    y_train = [2.0, 4.0, 6.0]

    print('Predict (before training)\t{}\t{:.4f}'.format(4, forward(4).item()))
    print('-*----------Progress Begin----------*-')
    for epoch in range(100):
        for x, y in zip(x_train, y_train):
            loss_epoch = loss(x, y)
            loss_epoch.backward()
            print('\tgard:{}\t{:.4f}\t{:3.4f}'.format(x, y, w.grad.item()))  # item 讲梯度值变为标量
            w.data = w.data - 0.01 * w.grad.data  # 权重更新是要用标量（tensor张量转为标量）
            w.grad.data.zero_()
        print('Predict (Progress:) epoch={:2d} loss={:3.4f}'.format(epoch, loss_epoch.item()))
    print('-*----------Progress End----------*-')
    print('Predict (after training)\t{}\t{:.4f}'.format(4, forward(4).item()))


if __name__ == '__main__':
    main()
