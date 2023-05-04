# @Time    :2021/11/17 11:38
# @Author  :CWP
# @FileName: MSNNTrain.py

import numpy as np
from sklearn.model_selection import train_test_split, KFold,StratifiedKFold
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import  os
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

##伪编码的参数
##pre-epoch 10
PRE_EPOCH = 30
T1_EPOCH = 30
T2_EPOCH = 50
A = 0.01

train_Batch_Size = 24

# batch size 为128
BATCH_SIZE = 120


use_gpu =  torch.cuda.is_available()   #判断GPU是否存在可用
h5file = "D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/sleepdata/h5/"
files = os.listdir(h5file)  # 得到文件夹下的所有文件名称
files_len = len(files)

savepath = "./result_kfold_5/"
##K折交叉验证的数量
kfold = 5
keys = ['F4-A1', 'C4-A1', 'O2-A1', 'EOG-L', 'EOG-R', 'EMG', 'label']

result_path = "D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/"

##scaler保存
scaler_path = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/scaler/'
scaler_files = os.listdir(scaler_path)

##pseudo文件保存
savepseudofile = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/pseudodata/'


##tensorboard
tensorboard_savepath = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/tensorboard/'
##tensorboard SavePath
writer = SummaryWriter(tensorboard_savepath + 'experiment_1')


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
    labels = (torch.from_numpy(labels)).type('torch.LongTensor')

    return data, labels

###归一化，并返回数据和训练的standardscaler
def standardScalerData(standardscaler, x_data):
    standardscaler.fit(x_data)
    x_standard = standardscaler.transform(x_data)
    return torch.from_numpy(x_standard), standardscaler


###训练模型
def trainModle(modlename, lossname, optimizername, data_loader, EPOCH, index):
    running_loss = 0.0
    standardscaler = StandardScaler()
    modlename.train()
    for epoch in range(EPOCH):


        all_train_label = []
        all_train_pred = []
        for step, (train_x, train_y) in enumerate(data_loader):  # 设定训练数据

            train_x, train_y = noshuffleData(train_x, train_y)
            ##归一化
            train_x, standardscaler = standardScalerData(standardscaler, train_x)

            ##将模型、数据和损失函数放进GPU
            if use_gpu:

                train_x = torch.unsqueeze(train_x, 1).type(torch.FloatTensor).cuda()
                train_y = torch.squeeze(train_y).type(torch.LongTensor).cuda()

            output = modlename(train_x)  #先 为数据加上通道维度，并改为float类型

            loss_train = lossname(output, train_y)
            # cross entropy loss

            optimizername.zero_grad()  # clear gradients for this training step
            loss_train.backward()  # backpropagation, compute gradients
            optimizername.step()

            train_pred_y = torch.max(output, 1)[1].cpu()
            all_train_pred.extend(train_pred_y)
            all_train_label.extend(train_y.cpu())

        # writer.add_scalar('kfold_' + str(index + 1)+ ':training loss ',
        #             loss_train / (step + 1),
        #             epoch)
        print("epoch: %d     loss: %.6f" % (epoch + 1, float(loss_train) / (step + 1)))
        running_loss = 0.0

        train_accuracy = accuracy_score(np.array(all_train_label).squeeze(), np.array(all_train_pred).squeeze())
        print("训练集准确率：", train_accuracy)
        #print("epoch: %d     loss: %.6f" % (epoch + 1, loss_train.cpu()))
        print("模型训练结束")


    # 保存标化器
    pickle.dump(standardscaler, open(scaler_path + "%s.pkl" % str(index + 1).zfill(4), 'wb'))

    torch.cuda.empty_cache()
    writer.close()
    torch.save(modlename.state_dict(), result_path + 'module/%s.pth' % str(index + 1).zfill(4))  #entire net
    return modlename, standardscaler

#####
###伪标签训练模型
def pseudoTrainModel(modlename, lossname, optimizername, data_loader, pseudo_x_all, standardscaler, PRE_EPOCH, EPOCH, index, x_test, y_test):
    max_accuracy = 0
    max_model = modlename
    for epoch in range(PRE_EPOCH, EPOCH):

        ##获取伪标签时，会将msnn改为model.eval()
        pseudo_data_loader = getLabelFromExcel(modlename, index, pseudo_x_all)

        modlename.train()

        all_pseudo_pred = []
        all_pseudo_label = []
        running_loss = 0.0
        data_len_all = 0
        for step, (train_data, pseudo_train_data) in enumerate(zip(data_loader, pseudo_data_loader)):  # 设定训练数据

            ##从data_loader中获取x,y的数据
            train_x = train_data[0]
            train_y = train_data[1]
            pseudo_x = pseudo_train_data[0]
            pseudo_y = pseudo_train_data[1]

            train_x, train_y = noshuffleData(train_x, train_y)
            pseudo_x, pseudo_y = noshuffleData(pseudo_x, pseudo_y)

            ##对训练数据和伪标签数据进行拼接
            train_pseudo_x = torch.cat((train_x, pseudo_x), dim=0)

            if use_gpu:  ###将模型和数据加入标签
                train_pseudo_x, standardscaler = standardScalerData(standardscaler, train_pseudo_x)
                train_pseudo_x = torch.unsqueeze(train_pseudo_x, 1).type(torch.FloatTensor).cuda()#先 为数据加上通道维度，并改为float类型

                train_y = torch.squeeze(train_y).type(torch.LongTensor).cuda()
                pseudo_y = torch.squeeze(pseudo_y).type(torch.LongTensor).cuda()
                modlename.cuda()
                lossname.cuda()

            output = modlename(train_pseudo_x)

            a = A * (epoch + 1 - PRE_EPOCH) / (T2_EPOCH - T1_EPOCH)
            if epoch < 50 :
                loss_train = lossname(output[0:train_x.shape[0]], train_y) + lossname(output[train_x.shape[0]:output.shape[0]], pseudo_y) * a
            else:
                loss_train = lossname(output[0:train_x.shape[0]], train_y) + lossname(output[train_x.shape[0]:output.shape[0]], pseudo_y) * A
            running_loss += float(loss_train)
            # cross entropy loss

            optimizername.zero_grad()  # clear gradients for this training step
            loss_train.backward()  # backpropagation, compute gradients
            optimizername.step()


            # train_pred_y = torch.max(F.softmax(output, dim=1), 1)[1].cpu()
            # all_pseudo_pred.extend(train_pred_y)
            # all_pseudo_label.extend(train_pseudo_y)
            # if step % 1000 == 0:    # every 1000 mini-batches...
            #
            # # ...log the running loss
            #
            #     writer.add_scalar('kfold_' + str(index + 1)+ ':training loss ',
            #                 running_loss / 1000,
            #                 epoch * len(data_loader) + step)

                # running_loss = 0.0

        print("epoch: %d     loss: %.6f" % (epoch + 1, running_loss / (step + 1)))


        #测试集训练
        torch_dataset_test = Data.TensorDataset(x_test, y_test.squeeze())
        data_loader_test = Data.DataLoader(dataset=torch_dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=1, drop_last=True)  ##num_workers是加载文件的核心数
        # scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
        modlename.eval()

        with torch.no_grad():
            pred_y_list = []
            label_y_list = []
            for step, (test_x, test_y) in enumerate(data_loader_test):  # 设定训练数据
            #     if (use_gpu):

                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
                modlename = modlename.cuda()
                output = modlename(test_x)
                test_pred_y = torch.max(F.softmax(output, dim=1), 1)[1].cpu()
                pred_y_list.extend(test_pred_y)
                label_y_list.extend(test_y)
            test_accuracy = accuracy_score(np.array(label_y_list).squeeze(), np.array(pred_y_list).squeeze())
        print("测试集准确率", test_accuracy)



        # train_accuracy = accuracy_score(np.array(all_pseudo_label).squeeze(), np.array(all_pseudo_pred).squeeze())
        # print("训练集准确率：", train_accuracy)
        # if test_accuracy > max_accuracy:
        #     max_accuracy = test_accuracy
        #     max_model = modlename

    torch.save(modlename.state_dict(), result_path + 'module/%d.pth' % (index + 1))  # entire net

    # 保存标化器
    pickle.dump(standardscaler, open(scaler_path + "%s.pkl" % str(index + 1), 'wb'))
    return max_model, standardscaler

def dataOverSampling(x_train, y_train):
    oversampling_x = []
    oversampling_y = []
    step = 0
    for y_train_batch in y_train:
        x_train_batch = x_train[step]
        series = CounterValue(y_train_batch.numpy().reshape(-1))

        if 1 in series.index or 0 in series.index:
            oversampling_x.append(x_train_batch.numpy())  ##先将tensor转为numpy，再储存在list中，以便后续可以转为tensor
            oversampling_y.append(y_train_batch.numpy())  ##先将tensor转为numpy，再储存在list中，以便后续可以转为tensor

        step += 1
    tensor_oversampling_x = torch.Tensor(oversampling_x)
    tensor_oversampling_y = torch.Tensor(oversampling_y)
    return torch.cat((x_train, tensor_oversampling_x), dim=0), torch.cat((y_train, tensor_oversampling_y), dim=0)

##使用MSNN模型获取伪标签
def getPseudoLabel(msnn, index, x_pseudo):
    msnn.eval()
    scaler_files = os.listdir(scaler_path)
    scaler = pickle.load(open(scaler_path + scaler_files[index], 'rb'))
    torch_dataset_pseudo_train = Data.TensorDataset(x_pseudo)
    pseudo_data_loader = Data.DataLoader(dataset=torch_dataset_pseudo_train, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    if use_gpu:
        with torch.no_grad():
            pseudo_y_list = []
            for step, (pseudo_x) in enumerate(pseudo_data_loader):  # 设定训练数据
            #     if (use_gpu):
                pseudo_x = pseudo_x[0]
                x_pseudo_standard = scaler.transform(pseudo_x)
                x_pseudo_standard = torch.from_numpy(x_pseudo_standard)
                pseudo_x = torch.unsqueeze(x_pseudo_standard, 1).type(torch.FloatTensor).cuda()
                msnn = msnn.cuda()
                output = msnn(pseudo_x)
                pseudo_pred_y = torch.max(F.softmax(output, dim=1), 1)[1].cpu()
                pseudo_y_list.extend(pseudo_pred_y)

            ##保存pseudo的label数据
            pseudo_y_list = np.array(pseudo_y_list, dtype='int32').reshape(1, -1)  # 转换类型
            # saveExcelFile(savepseudofile + str(index + 1) + ".csv", pseudo_y_list)
    return pseudo_y_list

#从excel表中获取伪标签
def getLabelFromExcel(msnn, index, pseudo_x):

        ##获取伪标签
        pseudo_y = getPseudoLabel(msnn, index, pseudo_x)
        # pseudo_files = os.listdir(savepseudofile)  # 得到文件夹下的所有文件名称
        # df = pd.read_csv(savepseudofile + pseudo_files[index], header=None)
        # pseudo_y = np.array(df, dtype='int32')

        pseudo_x, pseudo_y = cutData(pseudo_x[:pseudo_y.shape[1], :], pseudo_y, size=5)
        pseudo_x, pseudo_y = shuffleData(pseudo_x, pseudo_y, size=5)
        torch_dataset_pseudo_train = Data.TensorDataset(pseudo_x, torch.from_numpy(pseudo_y).squeeze())
        pseudo_data_loader = Data.DataLoader(dataset=torch_dataset_pseudo_train, batch_size=train_Batch_Size // 3 * 2, shuffle=True, drop_last=True)

        return pseudo_data_loader

#测试模型效果
def testModel(msnn, x_test, y_test, standardscaler, index):
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
                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
                msnn = msnn.cuda()
                output = msnn(test_x)
                test_pred_y = torch.max(F.softmax(output, dim=1), 1)[1].cpu()
                pred_y_list.extend(test_pred_y)
                label_y_list.extend(test_y)
            print("预测维度", np.array(pred_y_list).shape)
            test_accuracy = accuracy_score(np.array(label_y_list).squeeze(), np.array(pred_y_list).squeeze())
        print("测试集准确率", test_accuracy)
        ##保存测试集数据
        kappa, classification_report_result = cm_plot(np.array(label_y_list).squeeze(), np.array(pred_y_list).squeeze(),
                 "_kfold-%s" % (str(index + 1)), "%s/matrix/" % (result_path))
        np.savetxt('%s/kappa/kappa_%s.txt' % (result_path, str(index + 1)), [kappa])
        saveExcelFile("%s/report/report_%s.csv" % (result_path, str(index + 1)), classification_report_result)
        torch.cuda.empty_cache()

#测试模型效果
def testModelForEach(msnn, x_test, y_test, standardscaler, index, result_path):
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



                x_test_standard = standardscaler.transform(test_x)
                x_test_standard = torch.from_numpy(x_test_standard)
                test_x = torch.unsqueeze(x_test_standard, 1).type(torch.FloatTensor).cuda()
                msnn = msnn.cuda()
                output = msnn(test_x)
                test_pred_y = torch.max(F.softmax(output, dim=1), 1)[1].cpu()
                pred_y_list.extend(test_pred_y)
                label_y_list.extend(test_y)
            print("预测维度", np.array(pred_y_list).shape)
            test_accuracy = accuracy_score(np.array(label_y_list).squeeze(), np.array(pred_y_list).squeeze())
        print("测试集准确率", test_accuracy)
        ##保存测试集数据
        kappa, classification_report_result = cm_plot(np.array(label_y_list).squeeze(), np.array(pred_y_list).squeeze(),
                 "_kfold-%s" % (str(index + 1)), "%s/matrix/" % (result_path))
        print(kappa, classification_report_result)
