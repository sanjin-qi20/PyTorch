"""
============================
# -*- coding: utf-8 -*-
# Time    : 2022/10/11 16:31
# Author  : Qisx
# FileName: LinearModel.py
# Software: PyCharm
===========================
"""

import matplotlib.pyplot as plt
import numpy as np


def ForwardModel(x: float, w: float):
    """
    前推线性模型，w为权重
    """
    return x * w


def loss(y_train: float, y_predict: float):
    """
    计算训练值与预测结果的误差
    :param y_train: 训练集实际值
    :param y_predict: 训练集预测值
    :return: 误差的平方和
    """
    return pow((y_predict - y_train), 2)


def Gradient4Loss(x_train: float, y_predict: float, w: float):
    """
    :param x_train: 样本x值
    :param y_predict: 样本y值
    :param w: 前推模型权值，用于人工智能训练
    :return: 样本数据在此w值时的梯度
    """
    return 2 * x_train * (x_train * w - y_predict)


def cost(x_train: list, y_train: list, w: float):
    """
    计算整个数据集在模型预测结果与真实值的平均误差平方和
    :param x_train:训练集x
    :param y_train:训练集y
    :param w:前推模型权值，用于人工智能训练
    :return:预测结果与真实值的平均误差平方和
    """
    TraindataLength = len(x_train)
    cost = 0
    for x_train_in, y_train_in in zip(x_train, y_train):
        y_train_in_predict = ForwardModel(x_train_in, w)
        cost = loss(y_train_in, y_train_in_predict)
    return cost / TraindataLength


def Gradient4Cost(x_train: list, y_train: list, w: float):
    """
    计算当前w下的cost梯度
    :param x_train:训练集x
    :param y_train:训练集y
    :param w:前推模型权值，用于人工智能训练
    :return:用于修正权值w的梯度值
    """
    traindata_length = len(x_train)
    gard = 0
    for x, y in zip(x_train, y_train):
        gard += 2 * x * (x * w - y)
    return gard / traindata_length


# 梯度下降法实例
def GradientDescent():
    x_train = [1.0, 2.0, 3.0, ]
    y_train = [2.0, 4.0, 6.0]
    epoch_list = []
    cost_list = []
    w = 1
    learn_rate = 0.01
    print("epoch={:2}\tPredict (Before Training)\tw={}\tx={}\ty={}".format(0, w, 4, ForwardModel(4, w)))
    for epoch in range(100):
        cost_val = cost(x_train, y_train, w)
        gard_cal = Gradient4Cost(x_train, y_train, w)
        w -= gard_cal * learn_rate
        print("epoch={:2}\tPredict (Before Training)\tw={:.4f}\tx={}\ty={:.4f}".format(epoch + 1, w, 4,
                                                                                       ForwardModel(4, w)))
        epoch_list.append(epoch)
        cost_list.append(cost_val)
    plt.figure()
    plt.plot(epoch_list, cost_list)
    plt.xlabel("Epoch")
    plt.ylabel("COST")
    plt.show()


def StochasticGradientDescent():
    x_train = [1.0, 2.0, 3.0, ]
    y_train = [2.0, 4.0, 6.0]
    w = 1.0
    learn_rate = 0.01
    epoch_list = []
    loss_list = []
    print("epoch={:2}\tPredict (Before Training)\tw={:.4f}\tloss=None\tx={}\ty={:.4f}".format(0, w, 4,
                                                                                              ForwardModel(4, w)))
    for epoch in range(100):
        loss_epoch = 0
        for x, y in zip(x_train, y_train):
            gard = Gradient4Loss(x, y, w)
            w = w - gard * learn_rate
            loss_epoch = loss(y, ForwardModel(x, w))
        print("epoch={:2}\tPredict (Before Training)\tw={:.4f}\tloss={:.4f}\tx={}\ty={:.4f}".format(epoch + 1, w,
                                                                                                    loss_epoch, 4,
                                                                                                    ForwardModel(4, w)))
        epoch_list.append(epoch)
        loss_list.append(loss_epoch)
    plt.figure()
    plt.plot(epoch_list, loss_list)
    plt.xlabel("Epoch")
    plt.ylabel("COST")
    plt.show()
def main():
    # GradientDescent()
    StochasticGradientDescent()
    pass


if __name__ == '__main__':
    main()
