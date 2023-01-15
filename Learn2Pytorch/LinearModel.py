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


def loss(y_train: float, y_predict: float):
    """
    :param y_train: 实际值
    :param y_predict: 预测值
    :return: 误差的平方和
    """
    return pow((y_predict - y_train), 2)


def main():
    # 前推线性模型，w为权重
    ForwardModel = lambda x: x * w
    #
    x_train = [1.0, 2.0, 3.0]
    y_train = [2.0, 4.0, 6.0]

    g_w_list = []
    g_mse_list = []

    # 穷举法寻找w的最优解
    for w in np.arange(0, 4, 0.01):
        print('w=', w)
        loss_sum = 0
        for x_test, y_test in zip(x_train, y_train):
            y_predict_test = ForwardModel(x_test)
            loss_test = loss(y_test, y_predict_test)
            loss_sum = loss_sum + loss_test
            print('\t', x_train, y_train, x_test, y_predict_test)
        # 当误差下雨特定值是退出穷举
        # if loss_sum<1e-16:
        #     break
        print('MSE=', loss_sum)
        g_w_list.append(w)
        g_mse_list.append(loss_sum / 3)

    plt.figure()
    plt.plot(g_w_list, g_mse_list)
    plt.ylabel('Loss')
    plt.xlabel("w")
    plt.show()


if __name__ == '__main__':
    main()
