# -*- coding: utf-8 -*-
# @Time    : 2022/11/20 14:56
# @Author  : Qisx
# @File    : PlotData.py
# @Software: PyCharm

import numpy as np
from matplotlib import pyplot as plt


def y(x):
    # return
    # return np.log10(x)
    # return 106.3013 + 1.9714 * (x) ** 0.5
    # return 106.31477+3.9466*np.log10(x)
    return 111.4875-9.8333/x


xx = np.arange(1, 20, 0.01)
yx = y(xx)
print(y)

x = [2, 3, 4, 5, 7, 8, 10, 11, 14, 15, 16, 18, 19]
y = [106.4200, 108.2000, 109.5800, 109.5000, 110.0000, 109.9300, 110.4900, 110.5900, 110.6000, 110.9000, 110.7600,
     111.0000, 111.2000, ]

plt.figure()
plt.plot(xx,yx, 'black')
plt.grid(linestyle='--')
plt.scatter(x, y,color='red')
# plt.title(r'$ y = a+b\sqrt{x} $',size=20)
# plt.title(r'$ y = a+blog_{10}(x)$',size=20)
plt.title('model three', size=20)
# plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
# plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
# # plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.3))
# # plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.15))
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(1))
# plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.5))
plt.tick_params(which='major', length=10, width=2)
plt.tick_params(which='minor', length=4)
plt.tick_params(which='both', labelsize=18, right=False, left=True, top=False, bottom=True, direction='out')

plt.show()
