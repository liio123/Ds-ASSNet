# @Time    :2021/11/17 11:37
# @Author  :CWP
# @FileName: MSNNMainFunction.py
import numpy as np
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import os
from MSNN import MSNN

import torch
import torch.nn as nn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
import h5py
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, classification_report
import torch.nn.functional as F
from ResultCode import *
##from Attention import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from UtilsForDH import *
from FocalLoss import *
from MSNNTrain import *

##迭代次数为100
EPOCH = 50

##伪编码的参数
##pre-epoch 10
PRE_EPOCH = 30
T1_EPOCH = 30
T2_EPOCH = 50
A = 0.01


# batch size 为128
BATCH_SIZE = 24
##学习率为0.01
LEARNING_RATE1 = 0.0001
LEARNING_RATE2 = 0.0001
##动量因子为0.9
MOMENTUM = 0.8
##mlp隐藏单元
mlp_hsize = 300
##rnn隐藏单元
rnn_hsize =300

use_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
h5file = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/sleepdata/h5/'
files = os.listdir(h5file)  # 得到文件夹下的所有文件名称
files_len = len(files)
##K折交叉验证的数量
kfold = 5

model_path = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/module - transform/'
models = os.listdir(model_path)  # 得到文件夹下的所有文件名称

scaler_path = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/scaler/'
scalers = os.listdir(scaler_path)  # 得到文件夹下的所有文件名称
#
# del_data = [4, 12, 13, 17, 22, 23, 24]

channel_index = 5

channel_name = ['C3_M2', 'FP1_O1', 'EOG1', 'label']
# EOG horizontal  EMG submental

if __name__ == '__main__':

    kf = KFold(n_splits=5)
    index = 0
    print("开始读取数据")
    for train_index, test_index in kf.split(files):
        x_train, y_train = getEEGData(h5file, files, train_index, channel=channel_index)
        print("开始模型训练")
        Kappa = []
        print("%d折交叉验证--------" % (index + 1))
        msnn = MSNN()
        #msnn.load_state_dict(torch.load(model_path + models[1]))
        # standardscaler = pickle.load(open(scaler_path + scalers[1], 'rb'))
        optimizer = torch.optim.Adam(msnn.parameters(), lr=LEARNING_RATE1)
        msnn.train()

        # 将数据按照一个batch进行rehsape，方便后面打乱
        x_train_cut, y_train_cut = cutData(x_train, y_train, size=5)
        x_train_shuffle, y_train_shuffle = shuffleData(x_train_cut, y_train_cut, size=5)
        # 权重
        samples = CounterValue(y_train.numpy().reshape(-1))
        weights = torch.tensor([samples[0], samples[1], samples[2], samples[3], samples[4]], dtype=torch.float32)
        weights = weights / weights.sum()
        weights = 1.0 / weights
        weights = weights / weights.sum()
        print(weights)
        class_weight = torch.FloatTensor(weights).cuda()


        loss_func = nn.CrossEntropyLoss(weight=class_weight)
        x_train_shuffle_smapling, y_train_shuffle_smapling = dataOverSampling(x_train_shuffle, y_train_shuffle)
        print(CounterValue(y_train_shuffle_smapling.numpy().reshape(-1)))
        torch.cuda.empty_cache()
        if (use_gpu):
            msnn = msnn.cuda()
            loss_func = loss_func.cuda()
            # num_workers是加载文件的核心数
            torch_dataset_train = Data.TensorDataset(x_train_shuffle_smapling, y_train_shuffle_smapling.squeeze())
            data_loader = Data.DataLoader(dataset=torch_dataset_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)  ##num_workers是加载文件的核心数torch_dataset_train = Data.TensorDataset(x_train, y_train.squeeze())
            # 只包含带有真实标签的训练集，对模型进行训练
            msnn, standardscaler = trainModle(msnn, loss_func, optimizer, data_loader, EPOCH, index)
            x_test, y_test = getEEGData(h5file, files, test_index, channel=channel_index)
            testModel(msnn, x_test, y_test, standardscaler, index)
        index += 1






