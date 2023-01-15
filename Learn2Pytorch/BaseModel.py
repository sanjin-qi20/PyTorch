"""
============================
# -*- coding: utf-8 -*-
# Time    : 2022/10/17 15:41
# Author  : Qisx
# FileName: BaseModel.py
# Software: PyCharm
===========================
"""
import torch
from torch.nn import Module, Linear, MSELoss
from torch.optim import SGD

# Step1. Prepare Dataset
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# Step2. Design model using class
class LinearModel(Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = Linear(1, 1)

    def forward(self, x):  # 前推
        y_predict = self.linear(x)
        return y_predict


model = LinearModel()

# Step3. Construct loss and optimizer
criterion = MSELoss(size_average=False)
optimizer = SGD(model.parameters(), lr=0.05)

# Step Training Cycle
for epoch in range(120):
    y_predict = model(x_data)
    loss = criterion(y_predict, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.tensor([[4.0],[ 5.0],[ 6.0]])
y_test = model(x_test)
print('y_predict=', y_test.data)
