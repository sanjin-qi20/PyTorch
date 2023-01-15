"""
============================
# -*- coding: utf-8 -*-
# Time    : 2022/10/25 15:28
# Author  : Qisx
# FileName: ConvolutionNeuralNetworkBase.py
# Software: PyCharm
===========================
"""
import time

import torch
from matplotlib import pyplot as plt
# ------------Introducing library functions-------------
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from torch.nn import Module, Conv2d, MaxPool2d, Linear
from torch.nn.functional import relu

from torch.nn import CrossEntropyLoss
from torch.optim import SGD
# prepare dataset
start_time = time.perf_counter()
batch_size = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='Datasets/mnist',
                               train=True,
                               download=False,
                               transform=transform
                               )
train_dataloader = DataLoader(dataset=train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=2
                              )
test_dataset = datasets.MNIST(root='Datasets/mnist',
                              train=False,
                              download=False,
                              transform=transform)
test_dataloader = DataLoader(dataset=test_dataset,
                             shuffle=True,
                             batch_size=batch_size,
                             num_workers=2
                             )


# ---------design model using class-----------------
class CNNet(Module):
    def __init__(self):
        super(CNNet, self).__init__()
        self.conv1 = Conv2d(1, 10, kernel_size=5)
        self.conv2 = Conv2d(10, 20, kernel_size=5)
        self.pool_max = MaxPool2d(kernel_size=2)
        self.linear1 = Linear(320, 256)
        self.linear2 = Linear(256, 64)
        self.linear3 = Linear(64, 10)

    def forward(self, inputs):
        batch_size_ = inputs.size(0)
        x = relu(self.conv1(inputs))  # (1,28,28) to (10,24,24)
        x = self.pool_max(x)  # (10,24,24) to (10,12,12)
        x = relu(self.conv2(x))  # (10,12,12) to (20,8,8)
        x = self.pool_max(x)  # (20,8,8) to (20,4,4)
        x = x.view(batch_size_, -1)  # (20,4,4) to (64,5(自动计算))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


model = CNNet()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ---------construct loss and optimizer---------
criterion = CrossEntropyLoss(reduction='mean')
optimizer = SGD(model.parameters(),
                lr=0.001,
                momentum=0.5)


# ---------------training cycle------------
def train(epoch):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        y_predict = model(inputs)

        loss = criterion(y_predict, target)
        loss.to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 300 == 299:
            print('[{} \t {}] loss:{:2.4f}'.format(epoch + 1, i + 1, running_loss / 300))
            running_loss = 0.0


# ----------------testing cycle------------------
def test():
    corrct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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
    for epoch in range(5):
        epoch_list.append(epoch)
        train(epoch)
        # if epoch % 10 ==9:
        test()

    plt.plot(epoch_list, accuracy_list)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    end_time = time.perf_counter()
    print(end_time - start_time)
