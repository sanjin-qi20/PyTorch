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


class BackPropagation():
    w = torch.Tensor([1.0])
    w.requires_grad = True

    def forward(self, x):
        """
        前推模型
        :param x:训练集x值
        :return:预测值
        """
        return x * self.w

    def loss(self, x, y):
        """
        计算训练值与预测结果的误差
        :param y: 训练集实际值
        :param y: 训练集预测值
        :return: 误差的平方和
        """
        y_predict = self.forward(x)
        return pow((y_predict - y), 2)

    def back_propagation(self):
        self.x_train = torch.tensor([1.0, 2.0, 3.0])
        self.y_train = torch.tensor([2.0, 4.0, 6.0])
        epoch_list = []
        cost_list = []
        print('Predict (before training)\t{}\t{:.4f}'.format(4, self.forward(4).item()))
        print('-*------------------Progress Begin------------------*-')
        for epoch in range(100):
            loss_epoch = 0
            for x, y in zip(self.x_train, self.y_train):
                loss_epoch = self.loss(x, y)
                loss_epoch.backward()
                print('\tgard:{}\t{:.4f}\t{:3.4f}'.format(x, y, w.grad.item()))  # item 讲梯度值变为标量
                w.data = w.data - 0.01 * w.grad.data  # 权重更新是要用标量（tensor张量转为标量）
                w.grad.data.zore_()
                # print('\tgard:{}\t{:.4f}\t{:3.4f}'.format(x, y, self.w.grad.item()))  # item 讲梯度值变为标量
                # self.w.data = self.w.data - 0.01 * self.w.grad.data  # 权重更新是要用标量（tensor张量转为标量）
                # self.w.grad.data.zore_()
            print('Predict (Progress:) epoch={:2d} loss={:3.4f}'.format(epoch, loss_epoch.item()))
        print('-*------------------Progress End------------------*-')
        print('Predict (after training)\t{}\t{:.4f}'.format(4, self.forward(4).item()))


if __name__ == '__main__':
    w = torch.Tensor([1.0])
    B = BackPropagation().back_propagation()
