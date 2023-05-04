import h5py
from MSNN import MSNN
import torch
import os

# 安装h5py库,导入
f = h5py.File('E:/应少飞/sleepedf/sc/data/0150.h5', 'r')
# 读取文件,一定记得加上路径
# for key in f.keys():
#     print(f[key].name)
# # 打印出文件中的关键字
#     print(f[key].shape)
# # 将key换成某个文件中的关键字,打印出某个数据的大小尺寸
#     print(f[key][:])
print(f.keys())

# modelPath = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/module/'
# model_files = os.listdir(modelPath)  # 得到文件夹下的所有文件名称
# model = MSNN()
# # print(model)
# model.load_state_dict(torch.load(modelPath + model_files[0]))
# print(model.dropout)