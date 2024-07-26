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

from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
from UtilsForDH import *
from FocalLoss import *

from torchstat import stat
from thop import profile
import time

PRE_EPOCH = 30
T1_EPOCH = 30
T2_EPOCH = 50
A = 0.1
train_Batch_Size = 24
# batch size 为128
BATCH_SIZE = 120
use_gpu = torch.cuda.is_available()   #判断GPU是否存在可用

savepath = "./result_kfold_5/"
##K折交叉验证的数量
kfold = 5
keys = ['F4-A1', 'C4-A1', 'O2-A1', 'EOG-L', 'EOG-R', 'EMG', 'label']

# 保存excel格式文件
def save_to_file(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()


def getEEGData(h5file, files, files_index, channel):

    data = np.empty(shape=[0, 3000])
    labels = np.empty(shape=[0, 1])
    for filename in files_index:
        with h5py.File(h5file + files[filename], 'r') as fileh5:
            data = np.concatenate((data, fileh5[keys[channel]][:]), axis=0)
            labels = np.concatenate((labels, fileh5[keys[6]][:]), axis=0)
            labels[np.where(labels == -1)] = 2

    data = (torch.from_numpy(data)).type('torch.FloatTensor')
    print("data:",data.shape)
    labels = (torch.from_numpy(labels)).type('torch.LongTensor')

    return data, labels
    # else:
    #     return eeg_data, labels.squeeze()

###归一化，并返回数据和训练的standardscaler
def standardScalerData(standardscaler, x_data):
    standardscaler.fit(x_data)
    x_standard = standardscaler.transform(x_data)
    return torch.from_numpy(x_standard), standardscaler

###训练集训练模型

#测试模型效果
def testModel(msnn, x_test, y_test, standardscaler, index, all_label, all_pred, result_path):
    #训练集测试  采用gpu测试
    msnn.eval()

    #测试集训练
    torch_dataset_test = Data.TensorDataset(x_test, y_test.squeeze())
    data_loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=True)  ##num_workers是加载文件的核心数

    # scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
    if use_gpu:
        with torch.no_grad():
            pred_y_list = []
            label_y_list = []
            for step, (test_x, test_y) in enumerate(data_loader_test):  # 设定训练数据
            #     if (use_gpu):
            #     flops, params = profile(msnn, inputs=(test_x, ))
            #     print('flops: ', flops, 'params: ', params)
            #     print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))





                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor)
                # flops, params = profile(msnn, inputs=(test_x, ))
                # print('flops: ', flops, 'params: ', params)
                # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
                # stat(msnn, (test_x, ))
                test_x = test_x.cuda()
                msnn = msnn.cuda()
                start_time = time.time()
                output = msnn(test_x)
                end_time = time.time()
                # print("time:", end_time - start_time)
                test_pred_y = torch.max(output, 1)[1].cpu()
                pred_y_list.extend(test_pred_y)
                label_y_list.extend(test_y)
            print("预测维度", np.array(pred_y_list).shape)
            test_accuracy = accuracy_score(np.array(label_y_list).squeeze(), np.array(pred_y_list).squeeze())
        print("测试集准确率", test_accuracy)
        all = []
        all.append(label_y_list)
        all.append(pred_y_list)
        all_label.extend(label_y_list)
        all_pred.extend(pred_y_list)

        # print(np.array(all).T.shape)
        ##保存测试集数据
        # saveLabelFile(result_path + "label/%d.csv" % (index + 1), np.array(all).T)
        # kappa, classification_report_result, cm = cm_plot_number(np.array(label_y_list).squeeze(), np.array(pred_y_list).squeeze(),
        #         "_kfold-%s" % (str(index + 1)), "%s/matrix/" % (result_path))
        # np.savetxt('%s/kappa/kappa_%s.txt' % (result_path, str(index + 1)), [kappa])
        cm_report = []
        # saveExcelFile("%s/matrix/%s.csv" % (result_path, str(index + 1)), cm)
        # saveExcelFile("%s/report/report_%s.csv" % (result_path, str(index + 1)), classification_report_result)
        torch.cuda.empty_cache()

        return all_label, all_pred

def PlotSleepPicture(model, standardscaler, x_test, y_test, savepath, filename, name):
    model.eval()
    # 测试集训练
    torch_dataset_test = Data.TensorDataset(x_test, y_test.squeeze())
    data_loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=120, shuffle=False, num_workers=1,
                                       drop_last=True)  ##num_workers是加载文件的核心数
    # scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
    if use_gpu:
        with torch.no_grad():
            pred_y_list = []
            label_y_list = []
            output_list = []
            for step, (test_x, test_y) in enumerate(data_loader_test):  # 设定训练数据
                #     if (use_gpu):

                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
                model = model.cuda()
                output = model(test_x)
                test_pred_y = torch.max(output, 1)[1].cpu()

                # print(test_pred_y)

                output_list.extend(output.cpu().detach().numpy())
                pred_y_list.extend(test_pred_y)
                label_y_list.extend(test_y)
            pred_y = np.array(pred_y_list)
            test_y = np.array(label_y_list)
            output = np.array(output_list)
    # tSNE_2D(output, test_y, "01234", name)
    # plot_sleepPicture(label_y_list, pred_y_list, savepath, filename)
    draw_PRAUC(test_y, output)
    # ROCAUC(test_y, output)


def viewTest(model, standardscaler, x_test, y_test):
    model.eval()
    # 测试集训练
    torch_dataset_test = Data.TensorDataset(x_test, y_test.squeeze())
    data_loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=120, shuffle=False, num_workers=1,
                                       drop_last=True)  ##num_workers是加载文件的核心数
    # scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
    if use_gpu:
        with torch.no_grad():
            pred_y_list = []
            label_y_list = []
            for step, (test_x, test_y) in enumerate(data_loader_test):  # 设定训练数据
                #     if (use_gpu):
                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
                model = model.cuda()
                output = model(test_x)
                test_pred_y = torch.max(output, 1)[1].cpu()
                pred_y_list.extend(test_pred_y)
                label_y_list.extend(test_y)

    return label_y_list, pred_y_list

def viewModelPic(model, standardscaler, x_test, y_test):
    model.eval()
    # 测试集训练


    torch_dataset_test = Data.TensorDataset(x_test, y_test)
    data_loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=120, shuffle=False, num_workers=1,
                                       drop_last=True)  ##num_workers是加载文件的核心数
    # scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
    if use_gpu:
        with torch.no_grad():
            filter_output_list1 = []
            filter_output_list2 = []
            for step, (test_x, test_y) in enumerate(data_loader_test):  # 设定训练数据

                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
                model = model.cuda()
                filter_output1, filter_output2 = model(test_x)
                filter_output_list1.extend(np.array(torch.sum(filter_output1.cpu(), dim=2)))
                filter_output_list2.extend(np.array(torch.sum(filter_output2.cpu(), dim=2)))
    # np.save("D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/all/filter1.npy", filter_output_list1)
    # np.save("D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/all/filter2.npy", filter_output_list2)

    return np.array(filter_output_list1).T, np.array(filter_output_list2).T
#
# def viewModelPic(model, standardscaler, x_test, y_test):
#     model.eval()
#     # 测试集训练
#
#
#     torch_dataset_test = Data.TensorDataset(x_test, y_test)
#     data_loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=12, shuffle=False, num_workers=1,
#                                        drop_last=True)  ##num_workers是加载文件的核心数
#     # scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
#     if use_gpu:
#         with torch.no_grad():
#             filter_output_list1 = []
#             filter_output_list2 = []
#             for step, (test_x, test_y) in enumerate(data_loader_test):  # 设定训练数据
#
#                 x_test_standard = standardscaler.transform(test_x)
#                 x_test_standard = torch.from_numpy(x_test_standard)
#                 test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
#                 model = model.cuda()
#                 filter_output = model(test_x)
#
#     return np.array(filter_output.cpu())


def viewLSTMPic(model, standardscaler, x_test, y_test):
    model.eval()
    # 测试集训练
    torch_dataset_test = Data.TensorDataset(x_test, y_test)
    data_loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=120, shuffle=False, num_workers=1,
                                       drop_last=True)  ##num_workers是加载文件的核心数
    # scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
    if use_gpu:
        with torch.no_grad():
            lstm_output_list = []
            for step, (test_x, test_y) in enumerate(data_loader_test):  # 设定训练数据

                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
                model = model.cuda()
                lstm_output = model(test_x)
                lstm_output_list.extend(np.array(lstm_output.cpu()))
    # print(lstm_output_list.shape)
    np.save("D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/all/lstm.npy",lstm_output_list)

    return np.array(lstm_output_list)
