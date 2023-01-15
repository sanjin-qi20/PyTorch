"""
============================
# -*- coding: utf-8 -*-
# Time    : 2022/10/22 9:54
# Author  : Qisx
# FileName: MultiClassification.py
# Software: PyCharm
===========================
"""
# -----------Introducing library functions-----------
import torch
from matplotlib import pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, dataset

from torch.nn import Module, Linear
from torch.nn.functional import relu

from torch.nn import CrossEntropyLoss
from torch.optim import SGD

# -------------prepare datasets--------------------
batch_size = 32
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 所有样本的均值、标准差
])
# transforms.Compose:将内部[]内的所有对象构成PIL的数据
# transforms.ToTensor:将图像由28*28的单通道转为1*28*28的多通道
train_dataset = datasets.MNIST(root='Datasets/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=64)
test_dataset = datasets.MNIST(root='Datasets/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=True,
                         batch_size=batch_size
                         )


# -------------design model using class--------------------
class NetClassification(Module):
    def __init__(self):
        super(NetClassification, self).__init__()
        self.linear1 = Linear(784, 512)
        self.linear2 = Linear(512, 256)
        self.linear3 = Linear(256, 128)
        self.linear4 = Linear(128, 64)
        self.linear5 = Linear(64, 32)
        self.linear6 = Linear(32, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 784)  # -1：电脑帮助计算，长度
        x = relu(self.linear1(x))
        x = relu(self.linear2(x))
        x = relu(self.linear3(x))
        x = relu(self.linear4(x))
        x = relu(self.linear5(x))
        x = self.linear6(x)
        return x


model = NetClassification()

# -------------construct loss and optimizer--------------------
# 使用交叉熵函数
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(),
                lr=0.01,
                momentum=0.5)  # 冲破局部变量的极小值


# -------------training cycle and testing--------------------
def train(epoch):
    running_loss = 0.0
    for batch_inx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward +update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_inx % 300 == 299:
            print('[%d,\t%5d]\t loss:%1.6f' % (epoch + 1, batch_inx + 1, running_loss / 300))
            running_loss = 0.0


# ------------------testing--------------------
def test():
    corrct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicts = torch.max(outputs.data, dim=1)
            # 返回最大值与最大值所在下标
            # dim：按维度返回，行：第一维度；列：第二维度
            # 下标对应结果
            total += labels.size(0)
            corrct += (predicts == labels).sum().item()  # 张量的比较运算
    accuracy = corrct / total
    accuracy_list.append(accuracy)
    print('Accuracy on test set:%f %%' % (100 * corrct / total))


if __name__ == '__main__':
    epoch_list = []
    accuracy_list = []
    for epoch in range(50):
        epoch_list.append(epoch)
        train(epoch)
        # if epoch % 10 ==9:
        test()

    plt.plot(epoch_list, accuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
