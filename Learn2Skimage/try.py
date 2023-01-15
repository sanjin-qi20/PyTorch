# -*- coding: utf-8 -*-
# @Time    : 2022/9/28 18:54
# @Author  : Qisx
# @File    : try.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from skimage.filters import sobel, prewitt,laplace,scharr
from skimage import io, color

file = "yiwei.jpg"
# file = "cuijinghui.jpg"
camera = io.imread(file, as_gray=True)
camera1 = io.imread(file)
hig, wei = camera.shape
# edges = sobel(camera)
# edges = prewitt(camera)
# edges = laplace(camera)
edges = scharr(camera)
for i in range(hig):
    for j in range(wei):
        if edges[i][j] >= 0.035:
            edges[i][j] = 0
        else:
            edges[i][j] = 1
plt.figure()
plt.subplot(121)
plt.imshow(camera1)
plt.subplot(122)
plt.imshow(edges, cmap='gray')
