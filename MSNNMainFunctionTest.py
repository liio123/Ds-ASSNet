import numpy as np
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import os
from MSNN import *
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
from MSNNTest import *
# import librosa
import time


##迭代次数为100
EPOCH = 50

##伪编码的参数
##pre-epoch 10
PRE_EPOCH = 30
T1_EPOCH = 30
T2_EPOCH = 50
A = 0.1


# batch size 为128
BATCH_SIZE = 24
##学习率为0.01
LEARNING_RATE = 0.0001
##动量因子为0.9
MOMENTUM = 0.8
##mlp隐藏单元
mlp_hsize = 300
##rnn隐藏单元
rnn_hsize =300


LEARNING_RATE = 0.0001
LEARNING_RATE1 = 0.00001
LEARNING_RATE2 = 0.0001

use_gpu = torch.cuda.is_available()   #判断GPU是否存在可用
h5file = 'E:/sleep/sleepdata/h5/'
files = os.listdir(h5file)  # 得到文件夹下的所有文件名称
files_len = len(files)
result_path = "result/"
##K折交叉验证的数量
kfold = 5

modelPath = './result/module/'
model_files = os.listdir(modelPath)  # 得到文件夹下的所有文件名称
model_files_len = len(model_files)

scaler_path = './result/scaler/'
scaler_files = os.listdir(scaler_path)  # 得到文件夹下的所有文件名称
scaler_files_len = len(scaler_files)

sleeppicturePath = "./result/sleeppicture"
feature_path = "./result/feature_picture"
lstm_picture = "./result/lstm_picture"

channel_name = ['C4-A1', 'ECG', 'label']

##排序
def sort_filterout(out):
    final_out = []
    stage_number = [0]
    stage_number_list = [0]
    for stage_index in range(out.shape[0]):
        for filter_index in range(out.shape[1]):
            if np.argmax(out[:, filter_index]) == stage_index:
                final_out.append(out[:, filter_index])
        print("阶段%d" % (stage_index), len(final_out))
        stage_number.append(len(final_out))
        stage_number_list.append(len(final_out) - stage_number_list[stage_index])
    return final_out, stage_number, stage_number_list[1:]

if __name__ == '__main__':
    # start_time = time.time()
    kf = KFold(n_splits=5)
    index = 0
    print("开始读取数据")

    all_label = []
    all_pred = []
    channel_index = 1

    # for train_index, test_index in kf.split(files):
    #     print(train_index, test_index)
    #     x_test, y_test = getEEGData(h5file, files, test_index, channel=channel_index)
    #     model = MSNN()
    #     model.load_state_dict(torch.load(modelPath + model_files[index]))
    #     standardscaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
    #     ##测试模型
    #     all_label, all_pred = testModel(model, x_test, y_test, standardscaler, index, all_label, all_pred, result_path)
    #
    #     index += 1
    # all = []
    # all.append(all_label)
    # all.append(all_pred)
    # ##保存测试集数据
    # saveLabelFile(result_path + "all/label.csv", np.array(all).T)
    # kappa, classification_report_result, cm = cm_plot_number(np.array(all_label).squeeze(),
    #                                                          np.array(all_pred).squeeze(),
    #                                                          "all",
    #                                                          "%s/all/" % (result_path))
    # np.savetxt('%sall/kappa.txt' % (result_path), [kappa])
    # cm_report = []
    # saveExcelFile("%sall/cm.csv" % (result_path), cm)
    # saveExcelFile("%sall/report.csv" % (result_path), classification_report_result)

    # 睡眠阶段分期图
    # for i in range(83):
    i = 1
    eeg_data, labels = getEEGData(h5file, files, [39, 40, 41], channel=channel_index)
    print(eeg_data.shape)
    model = MSNN()
    model.load_state_dict(torch.load(modelPath + model_files[1]))
    standardscaler = pickle.load(open(scaler_path + scaler_files[1], 'rb'))
    # start_time = time.time()
    testModel(model, eeg_data, labels, standardscaler, index, all_label, all_pred, result_path)
    # end_time = time.time()
    # print("time:", end_time-start_time)
    # PlotSleepPicture(model, standardscaler, eeg_data, labels, sleeppicturePath, "1_", i)
    # viewModelPic(model,standardscaler,eeg_data,labels)
    # viewLSTMPic(model,standardscaler,eeg_data,labels)


    # # 特征过滤器激活图
    # files_index = 0
    # model_index = 3
    # flag = 2
    # eeg_data, labels = getEEGData(h5file, files, [0], channel=channel_index)
    #
    # model = MSNN()
    # model.load_state_dict(torch.load(modelPath + model_files[model_index]))
    # standardscaler = pickle.load(open(scaler_path + scaler_files[model_index], 'rb'))
    # label_y_list, pred_y_list = viewTest(model, standardscaler, eeg_data, labels)
    # # same_index = np.arange(len(label_y_list))[np.array(label_y_list) == np.array(pred_y_list)]
    # # print(len(same_index))
    # # pred_y_list = np.array(pred_y_list)[same_index]
    #
    # stage_list_data = [[], [], [], [], []]
    # stage_list_y = [[], [], [], [], []]
    #
    # pic_list_data1 = []
    # pic_list_data2 = []
    #
    # for stage in range(5):
    #     pred_y_list = np.array(pred_y_list)
    #     stage_list_data[stage] = eeg_data[np.where(pred_y_list == stage)]
    #     stage_list_y[stage] = pred_y_list[np.where(pred_y_list == stage)]
    #     model = viewMSNN()
    #     save_model = torch.load(modelPath + model_files[1])
    #     model_dict = model.state_dict()
    #     state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    #     model_dict.update(state_dict)
    #     model.load_state_dict(model_dict)
    #     standardscaler = pickle.load(open(scaler_path + scaler_files[3], 'rb'))
    #     out1, out2 = viewModelPic(model, standardscaler, stage_list_data[stage], torch.from_numpy(stage_list_y[stage]))
    #     print(out1)
    #     print(out2.shape)
    #     out1 = np.sum(out1, axis=1) / (out1.shape[1])
    #     out2 = np.sum(out2, axis=1) / (out2.shape[1])
    #     from sklearn.preprocessing import MinMaxScaler
    #     import matplotlib.pyplot as pyplot
    #
    #
    #     pic_list_data1.append(MinMaxScaler().fit_transform(out1.reshape(-1, 1)))
    #     pic_list_data2.append(MinMaxScaler().fit_transform(out2.reshape(-1, 1)))
    # out1 = np.array(pic_list_data1).squeeze()
    # out2 = np.array(pic_list_data2).squeeze()
    # import seaborn as sns
    #
    # final_out2, stage_number, stage_number_list = sort_filterout(out2)
    # final_out2 = np.array(final_out2).T
    #
    # sns.heatmap(data=final_out2[0:1, 0:32], square=True, cbar=False, cmap="gist_rainbow")
    # plt.yticks([])
    # plt.xticks([])
    # plt.savefig("%s/%d_feature_picture_tick%d.svg" % (feature_path, flag, files_index + 1))
    # pyplot.show()
    #
    # sns.heatmap(data=np.array(final_out2).T, square=True, cbar=False, cmap="RdBu_r")
    # plt.yticks([])
    # plt.xticks([])
    # plt.savefig("%s/%d_feature_picture%d.svg" % (feature_path, flag, files_index + 1))
    # pyplot.show()
    # np.savetxt("%s/%d_feature_picture_number%d.txt" % (feature_path, flag, files_index + 1), stage_number)
    #
    # # LSTM可视化图
    # files_index = 0
    # model_index = 3
    # eeg_data, labels = getEEGData(h5file, files, [0], channel=channel_index)
    # # print(eeg_data,labels)
    # model = MSNN()
    # model.load_state_dict(torch.load(modelPath + model_files[model_index]))
    # standardscaler = pickle.load(open(scaler_path + scaler_files[model_index], 'rb'))
    # label_y_list, pred_y_list = viewTest(model, standardscaler, eeg_data, labels)
    # # same_index = np.arange(len(label_y_list))[np.array(label_y_list) == np.array(pred_y_list)]
    # # print(len(same_index))
    # # pred_y_list = np.array(pred_y_list)[same_index]
    #
    # pred_y_list = np.array(pred_y_list)
    # model = viewLSTM()
    # save_model = torch.load(modelPath + model_files[model_index])
    # model_dict = model.state_dict()
    # state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # model.load_state_dict(model_dict)
    # out = viewLSTMPic(model, standardscaler, eeg_data[:len(pred_y_list)], torch.from_numpy(pred_y_list))
    # out = np.sum(out, axis=1) / 5
    #
    # from sklearn.preprocessing import MinMaxScaler
    # import matplotlib.pyplot as pyplot
    # import seaborn as sns
    #
    #
    # out = out.reshape(-1, 40)
    # pred_y = []
    # for np_stage in pred_y_list:
    #     if np_stage == 0:
    #         pred_y.append('W')
    #     if np_stage == 1:
    #         pred_y.append('N1')
    #     if np_stage == 2:
    #         pred_y.append('N2')
    #     if np_stage == 3:
    #         pred_y.append('N3')
    #     if np_stage == 4:
    #         pred_y.append('R')
    #
    # pred_y_list = np.array(pred_y).reshape(-1, 40)
    #
    # sns.heatmap(data=out, square=True, cbar=False, cmap="viridis", fmt='', annot=pred_y_list, annot_kws={'size':6, 'color':'black'}) #GnBu
    # plt.yticks([])
    # plt.xticks([])
    # # plt.savefig("%s/_lstm_picture%d.svg" % (lstm_picture, files_index + 1))
    # pyplot.show()