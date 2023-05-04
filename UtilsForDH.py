import pyedflib
import numpy as np
import h5py
import csv
import math
import pandas as pd
import os
from sklearn.model_selection import StratifiedShuffleSplit
import scipy.io as scio
from scipy import signal
# from openpyxl import Workbook
dodh_h5file = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/sleepdata/h5_2/'
dodh_h5file_label = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/sleepdata/data/label2/'
dodh_raw = 'D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/sleepdata/data/raw/'
dodh_raw_files = os.listdir(dodh_raw)  #得到文件夹下的所有文件名称



channels = ['Fpz-Cz']
channel_names = [['abd', 'c3m2', 'c4m1', 'e1', 'e2', 'ecg1ecg2', 'emg1emg2', 'external_pressur', 'f3f4', 'f3m2', 'f3o1',
                  'f4m1', 'f4o2', 'fp1f3', 'fp1m2', 'fp1o1', 'fp2f4', 'fp2m1', 'fp2o2', 'o1m2', 'o2m1', 'pulse', 'snore'
                     , 'spo2', 'the', 'tho']]

##保存Excel文件，并指定小数位数
def saveLabelFile(file_name, contents):
    # 写
    df = pd.DataFrame(contents)
    df.to_csv(file_name, index=None, header=None)


def getLabel(annotations1, annotations2):
    ##获取数据对应的label
    row = 0
    if int(float(annotations1[row])) >= 2400:
        annotations1[row] = 2400
    #annotations1[-2] = 1800   ##截取睡眠结束后的半小时W期
    label = []
    annotations2error = []
    for annotation1 in annotations1[0:-1]:

        if annotations2[row] == 'Sleep stage W':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 0
        elif annotations2[row] == 'Sleep stage R':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 4
        elif annotations2[row] == 'Sleep stage 1':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 1
        elif annotations2[row] == 'Sleep stage 2':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 2
        elif annotations2[row] == 'Sleep stage 3':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 3
        elif annotations2[row] == 'Sleep stage 4':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 3
        elif annotations2[row] == 'Sleep stage ?':
            annotation1 = 0
            ##用来统计除了最后一个阶段为？之外，阶段未知的数据
            if row != (len(annotations1) - 1):
                annotations2error.append(row)
        else:
            annotation1 = 0
            annotations2error.append(row)
        for index in range(annotation1):
            label.extend(annotations2[row])
        row += 1

    return label, annotations2error

def getNoLabel(annotations1, annotations2):
    ##获取数据对应的label
    row = 0
    label = []
    annotations2error = []
    for annotation1 in annotations1[0:-1]:

        if annotations2[row] == 'Sleep stage W':
            annotation1 = 0
            annotations2[row] = 0
            annotations2error.append(row)
        elif annotations2[row] == 'Sleep stage R':
            annotation1 = 0
            annotations2[row] = 4
            annotations2error.append(row)
        elif annotations2[row] == 'Sleep stage 1':
            annotation1 = 0
            annotations2[row] = 1
            annotations2error.append(row)
        elif annotations2[row] == 'Sleep stage 2':
            annotation1 = 0
            annotations2[row] = 2
            annotations2error.append(row)
        elif annotations2[row] == 'Sleep stage 3':
            annotation1 = 0
            annotations2[row] = 3
            annotations2error.append(row)
        elif annotations2[row] == 'Sleep stage 4':
            annotation1 = 0
            annotations2[row] = 3
            annotations2error.append(row)
        elif annotations2[row] == 'Sleep stage ?':
            annotation1 = int(float(annotation1) // 30)
            annotations2[row] = 5
        else:
            annotation1 = 0
            annotations2error.append(row)
        for index in range(annotation1):
            label.extend(annotations2[row])
        row += 1

    return label, annotations2error

# 去除标签异常的数据
def deleteOtherData(datas, annotations2error, annotations0, annotations1):
    deletelist = []
    for index in annotations2error:

        delete_index = np.arange(int(float(annotations0[index])) * 100,
                                 (int(float(annotations0[index])) + int(float(annotations1[index]))) * 100,
                                 dtype=int)
        deletelist.extend(delete_index)

    datas = np.delete(datas, deletelist)

    return datas

# 判断数据点长度和个数是否一致
def judgementDataLen(datasecond, datasize):

    if int(float(datasecond)) * 100 == datasize:
        return True
    else:
        return False

##判断数据点长度和个数是否一致
def getTrueannotations1(annotations0, annotations1):
    for index in range(len(annotations0) - 1):
        annotations1[index] = int(float(annotations0[index + 1])) - int(float(annotations0[index]))
    annotations1[index + 1] = int(float(annotations1[index + 1]))
    return annotations1

###获取EDF文件中所需要通道的数据
def getNoLabelData(datafile, labelfile, savefile, savelabelfile, channelsNum, sample_rate, channelsData, labels):

    fileedf = pyedflib.EdfReader(datafile)
    filelabel = pyedflib.EdfReader(labelfile)
    signal_headers = fileedf.getSignalHeaders()
    channelsName = []

    filedataError = 0
    for index in range(channelsNum):

        channel_name = signal_headers[index]['label'][4:]
        channelsName.append(channel_name)

        #print("通道名称", channel_name)
        column = sample_rate * 30  # 一个30秒epoch的列数
        size = fileedf.readSignal(index, 0, None, False).size  # 获取数据点总数

        ##获取events
        annotations = np.array(filelabel.readAnnotations())
        if judgementDataLen(annotations[0][-1], size) != True:
            annotations[0][-1] = size // 100

        annotations[1] = getTrueannotations1(annotations[0], annotations[1])
        label, annotations2error = getNoLabel(annotations[1], annotations[2])
        if index == 0:
            labels.extend(label)

        ##防止第一个分期小于半个小时的持续时间
        if int(float(annotations[0][1]) * sample_rate) > 30 * 80 * sample_rate:
            start = int(float(annotations[0][1]) * sample_rate) - 30 * 80 * sample_rate
        else:
            start = int(float(annotations[0][0]))

        ##end = size - int(float(annotations[1][-2]) * 100) + 30 * 60 * sample_rate  ##将过多的W期数据截取
        end = size
        oldsignals = fileedf.readSignal(index, 0, end, False)
        if len(annotations2error) != 0:

            oldsignals = deleteOtherData(oldsignals, annotations2error, annotations[0], annotations[1])
        #print(annotations2error)
        signals = oldsignals

        channelsData[index].extend(signals.reshape(-1, column))
    return channelsData, labels, channelsName


###获取EDF文件中的数据
def getChannelData(datafile, labelfile, savefile, savelabelfile, channelsNum=1, sample_rate=200):

    print("labelfile = ",labelfile)
    fileedf = pyedflib.EdfReader(datafile)
    filelabel = pd.read_csv(labelfile)
    signal_headers = fileedf.getSignalHeaders()
    channelsName = ['F4-A1', 'C4-A1', 'O2-A1', 'EOG-L', 'EOG-R', 'EMG']
    channelsData = []
    for index in range(6):
        size = fileedf.readSignal(index, 0, None, False).size  # 获取数据点总数
        print(filelabel.size)
        end = filelabel.size * sample_rate * 30
        if end > size:
            end = size // sample_rate // 30 * sample_rate * 30
        oldsignals = fileedf.readSignal(index, 0, end, False)

        column = sample_rate * 30
        newsignals = oldsignals.reshape(-1, column)
        newsignals = signal.resample_poly(newsignals, 100, sample_rate, axis=1)
        channelsData.append(newsignals)
        print(newsignals.shape)

    with h5py.File(savefile, 'w') as fileh5:
        fileh5['sample_rate'] = np.array([sample_rate], dtype=int)
        for channelindex in range(len(channelsData)):
            fileh5[channelsName[channelindex]] = channelsData[channelindex]  # 存储数据点
        labels = np.array(filelabel, dtype='int32').reshape(-1, 1)  # 转换类型
        # saveLabelFile(savelabelfile, labels)
        fileh5['label'] = labels

#统计字符串中元素出现的次数
def CounterValue(data):
    return pd.value_counts(data)

def getWeight(dict, labelCount):
    weight = []

    weight.append(float(labelCount) / dict[0])
    weight.append(float(labelCount) / dict[1])
    weight.append(float(labelCount) / dict[2])
    weight.append(float(labelCount) / dict[3])
    weight.append(float(labelCount) / dict[4])

    return weight

def getData(rawfile, savefile, labelfile, channel_list):
    with h5py.File(rawfile, mode='r') as fileh5:
        print(fileh5.keys())
        labels = fileh5['hypnogram']
        signal_list = fileh5['signals']

        channelsName = ['C3_M2', 'FP1_O1', 'EOG1']
        channelsData = []
        import mne
        newsignal = mne.filter.resample(np.array(signal_list['eeg'][channelsName[0]], dtype='float'), down=2.5)

        channelsData.append(newsignal.reshape(-1, 3000))
        newsignal = mne.filter.resample(np.array(signal_list['eeg'][channelsName[1]], dtype='float'), down=2.5)
        channelsData.append(newsignal.reshape(-1, 3000))
        newsignal = mne.filter.resample(np.array(signal_list['eog'][channelsName[2]], dtype='float'), down=2.5)
        channelsData.append(newsignal.reshape(-1, 3000))

        with h5py.File(savefile, 'w') as fileh5:
            fileh5['sample_rate'] = np.array([100], dtype=int)
            for channelindex in range(len(channelsData)):
                fileh5[channelsName[channelindex]] = channelsData[channelindex]  # 存储数据点
            labels = np.array(labels, dtype='int32').reshape(-1, 1)  # 转换类型
            fileh5['label'] = labels

##将EDF文件中都数据提取并放入H5文件中

if __name__ == '__main__':

    file_indexs = np.array(range(len(dodh_raw)))
    file_num = 152
    channelsData = [[], [], []]
    labels = []
    print("开始提取数据-----")
    print(dodh_raw_files)
    for file_index in dodh_raw_files:
        getChannelData(dodh_raw + file_index, dodh_h5file_label +'AST'+ str(file_num) +'_Labels' + '.csv', dodh_h5file +
                str(file_num + 1).zfill(3) + '.h5', str(file_num + 1).zfill(3) + '.csv')
        file_num += 2
    print("提取数据结束-----")

def cutData(x_data, y_data, size):
    len = x_data.shape[0] // size * size
    x_data = x_data[:len, :]
    y_data = y_data[:len, :]
    return x_data, y_data

##将数据按照一个batch进行reshape
def shuffleData(x_data, y_data, size):
    x_data = x_data.reshape(-1, size, 3000)
    y_data = y_data.reshape(-1, size, 1)
    return x_data, y_data

##将数据按照一个batch反reshape
def noshuffleData(x_data, y_data):
    x_data = x_data.reshape(-1, 3000)
    y_data = y_data.reshape(-1, 1)
    return x_data, y_data

## 划分数据集
def split_dataset(files, split_size=0.2):
    list_1, list_2 = None, None
    labels = np.ones(shape=[len(files)])
    split = StratifiedShuffleSplit(n_splits=1, test_size=split_size, random_state=1)
    for index1, index2 in split.split(files, labels):
        list_1 = index1
        list_2 = index2
    return list_1, list_2

