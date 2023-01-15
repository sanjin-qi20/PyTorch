"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2022/10/11 16:31
# @Author  : Qisx
# @FileName: LinearModel.py
# @Software: PyCharm
===========================
"""

import torch

print('CUDA版本:', torch.version.cuda)
print('Pytorch版本:', torch.__version__)
print('显卡是否可用:', 'yes' if (torch.cuda.is_available()) else 'no')
print('显卡数量:', torch.cuda.device_count())
print('当前显卡的CUDA算力:', torch.cuda.get_device_capability(0))
print('当前显卡型号:', torch.cuda.get_device_name())
