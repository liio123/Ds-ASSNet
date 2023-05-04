import pyedflib
import numpy as np
import h5py
import csv
import math
import pandas as pd
import os

# from mne.io import concatenate_raws, read_raw_edf
# import matplotlib.pyplot as plt
# import mne
# raw = read_raw_edf(data_path + data_files[1], preload=False)
#
# events_from_annot, event_dict = mne.events_from_annotations(raw)
# print(event_dict)
# print(events_from_annot)
from openpyxl import Workbook

sc_data_path = '../SleepEdfData/sleep-cassette/'   # 存放数据的具体位置，需要改成自己数据存放的地方
sc_data_files = os.listdir(sc_data_path) #得到文件夹下的所有文件名称

st_data_path = '../SleepEdfData/sleep-telemetry/'   # 存放数据的具体位置，需要改成自己数据存放的地方
st_data_files = os.listdir(st_data_path) #得到文件夹下的所有文件名称

sc_h5file = "../SleepEdfData/SCDataSet/"
sc_h5_files = os.listdir(sc_h5file) #得到文件夹下的所有文件名称

st_h5file = "../SleepEdfData/STDataSet/"
st_h5_files = os.listdir(st_h5file) #得到文件夹下的所有文件名称

channels = ['Fpz-Cz']



# def saveExcelFile(datas, filename):  ###将文件保存为excel文件
#     # pip install openpyxl
#     pass
#     workbook = Workbook()
#
#     # 默认sheet
#     sheet = workbook.active
#     # sheet = workbook.create_sheet(title="新sheet")
#     for data in datas:
#         sheet.append(list(data))
#
#
#     workbook.save(filename +'.xlsx')

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

    #annotations1[-2] = 1800   ##截取睡眠结束后的半小时W期
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


##去除标签异常的数据
def deleteOtherData(datas, annotations2error, annotations0, annotations1):

    deletelist = []
    for index in annotations2error:

        delete_index = np.arange(int(float(annotations0[index])) * 100,
                                 (int(float(annotations0[index])) + int(float(annotations1[index]))) * 100,
                                 dtype=int)
        deletelist.extend(delete_index)

    datas = np.delete(datas, deletelist)
        #print(delete_index)


    return datas


##判断数据点长度和个数是否一致
def judgementDataLen(datasecond, datasize):

    if int(float(datasecond)) * 100 == datasize:
        return True
    else:
        return False

##判断数据点长度和个数是否一致
def getTrueannotations1(annotations0, annotations1):
    for index in range(len(annotations0) - 1):
        annotations1[index] = int(float(annotations0[index + 1])) - int(float(annotations0[index]))
        # if (int(float(annotations0[index + 1])) - int(float(annotations0[index]))) > int(float(annotations1[index])):
        #annotations1[index] = int(float(annotations0[index + 1])) - int(float(annotations0[index]))
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

        ##从数据文件中获取通道名称

        # if len(signal_headers[index]['label']) <= 10:
        #     channel_name = signal_headers[index]['label'][4:]
        # else:
        #     channel_name = signal_headers[index]['label']

        channel_name = signal_headers[index]['label'][4:]
        channelsName.append(channel_name)

        #print("通道名称", channel_name)
        column = sample_rate * 30  # 一个30秒epoch的列数
        size = fileedf.readSignal(index, 0, None, False).size  # 获取数据点总数

        ##获取events
        annotations = np.array(filelabel.readAnnotations())

        ##处理方法1 ：直接丢弃该被试数据
        # if judgementDataLen(annotations[0][-1], size):
        #     filedataError = True
        # else:
        #     filedataError = False

        ###处理方法2： 将annotations[0]的秒数，通过数据点个数来进行修改
        if judgementDataLen(annotations[0][-1], size) != True:
            annotations[0][-1] = size // 100

        annotations[1] = getTrueannotations1(annotations[0], annotations[1])
        label, annotations2error = getNoLabel(annotations[1], annotations[2])
        if index == 0:
            labels.extend(label)

        ##截取中间部分   从不是W分期开始，往前取半个小时的数据，将分期为？的分期数据舍去


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


    # if filedataError == True:
    #     with h5py.File(savefile, 'w') as fileh5:
    #         fileh5['sample_rate'] = np.array([sample_rate], dtype=int)
    #         for channelindex in range(len(channelsData)):
    #             fileh5[channelsName[channelindex]] = channelsData[channelindex].reshape(-1, column)  # 存储数据点
    #         print("测试", len(labels))
    #         labels = np.array(labels, dtype='int32').reshape(-1, 1)  # 转换类型
    #         print(labels.shape, channelsData[channelindex].shape)
    #         if labels.shape[0] != channelsData[channelindex].shape[0]:
    #             print("annotations", annotations[0], annotations[1])
    #         saveExcelFile(labels, savelabelfile)
    #         fileh5['label'] = labels

    return channelsData, labels, channelsName



        # with open(labelfile, "r") as f:
        #     csv_read = csv.reader(f)
        #     labels = []
        #     # print("行数",row)
        #     for labelsum, label in enumerate(csv_read):
        #         if labelsum != row:
        #             labels.append(label[0])  # 或许对应数量的标签
        #         else:
        #             break
        #     labels = np.array(labels, dtype='int32').reshape(-1, 1)  # 转换类型
        #     fileh5['labels'] = labels
        # fileedf.close()






###获取EDF文件中无标签的数据
def getChannelData(datafile, labelfile, savefile, savelabelfile, channelsNum, sample_rate):

    fileedf = pyedflib.EdfReader(datafile)
    filelabel = pyedflib.EdfReader(labelfile)
    signal_headers = fileedf.getSignalHeaders()
    channelsName = []
    channelsData = []
    filedataError = 0
    for index in range(channelsNum):

        ##从数据文件中获取通道名称

        # if len(signal_headers[index]['label']) <= 10:
        #     channel_name = signal_headers[index]['label'][4:]
        # else:
        #     channel_name = signal_headers[index]['label']

        channel_name = signal_headers[index]['label'][4:]
        channelsName.append(channel_name)

        #print("通道名称", channel_name)
        column = sample_rate * 30  # 一个30秒epoch的列数
        size = fileedf.readSignal(index, 0, None, False).size  # 获取数据点总数

        ##获取events
        annotations = np.array(filelabel.readAnnotations())

        ##处理方法1 ：直接丢弃该被试数据
        # if judgementDataLen(annotations[0][-1], size):
        #     filedataError = True
        # else:
        #     filedataError = False

        ###处理方法2： 将annotations[0]的秒数，通过数据点个数来进行修改
        if judgementDataLen(annotations[0][-1], size) != True:
            annotations[0][-1] = size // 100

        annotations[1] = getTrueannotations1(annotations[0], annotations[1])
        labels, annotations2error = getLabel(annotations[1], annotations[2])

        ##截取中间部分   从不是W分期开始，往前取半个小时的数据，将分期为？的分期数据舍去


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
        signals = oldsignals[start:]

        channelsData.append(signals.reshape(-1, column))


    # if filedataError == True:
    #     with h5py.File(savefile, 'w') as fileh5:
    #         fileh5['sample_rate'] = np.array([sample_rate], dtype=int)
    #         for channelindex in range(len(channelsData)):
    #             fileh5[channelsName[channelindex]] = channelsData[channelindex].reshape(-1, column)  # 存储数据点
    #         print("测试", len(labels))
    #         labels = np.array(labels, dtype='int32').reshape(-1, 1)  # 转换类型
    #         print(labels.shape, channelsData[channelindex].shape)
    #         if labels.shape[0] != channelsData[channelindex].shape[0]:
    #             print("annotations", annotations[0], annotations[1])
    #         saveExcelFile(labels, savelabelfile)
    #         fileh5['label'] = labels
    with h5py.File(savefile, 'w') as fileh5:
        fileh5['sample_rate'] = np.array([sample_rate], dtype=int)
        for channelindex in range(len(channelsData)):
            fileh5[channelsName[channelindex]] = channelsData[channelindex].reshape(-1, column)[:1080, :]  # 存储数据点

        labels = np.array(labels, dtype='int32').reshape(-1, 1)[:1080, :]  # 转换类型
        #print(labels.shape, channelsData[channelindex].shape)
        #if labels.shape[0] != channelsData[channelindex].shape[0]:
            #print(datafile)
            #print("annotations", annotations[0], annotations[1], annotations[2])
        saveLabelFile(savelabelfile, labels)
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


##将EDF文件中都数据提取并放入H5文件中

if __name__ == '__main__':

    file_indexs = np.array(range(len(sc_data_files)))[::2]
    file_num = 0
    channelsData = [[], [], []]
    labels = []

    print("开始提取数据-----")
    import random

    random.seed(2)
    random.shuffle(file_indexs)

    #提取sc的数据
    for file_index in file_indexs:
        getChannelData(sc_data_path + sc_data_files[file_index], sc_data_path + sc_data_files[file_index + 1], sc_h5file + "data/" + str(file_num + 1).zfill(4) + ".h5",
                       sc_h5file + "label/" + str(file_num + 1).zfill(4), 3, 100)

        file_num += 1
    print("提取数据结束-----")

    # # 提取sc的数据
    # for file_index in file_indexs:
    #     channelsData, labels, channelsName = getNoLabelData(sc_data_path + sc_data_files[file_index], sc_data_path + sc_data_files[file_index + 1],
    #                    sc_h5file + "data/" + str(file_num + 1).zfill(4) + ".h5",
    #                    sc_h5file + "label/" + str(file_num + 1).zfill(4), 3, 100, channelsData, labels)
    #
    #
    #
    #
    #     file_num += 1
    #
    #
    #
    # with h5py.File(sc_h5file + "NoLabelData/data/data.h5", 'w') as fileh5:
    #     fileh5['sample_rate'] = np.array([100], dtype=int)
    #     for channelindex in range(3):
    #
    #         fileh5[channelsName[channelindex]] = np.array(channelsData[channelindex]).reshape(-1, 3000)  # 存储数据点
    #
    #     labels = np.array(labels, dtype='int32').reshape(-1, 1)  # 转换类型
    #
    #     # print(labels.shape, channelsData[channelindex].shape)
    #     # if labels.shape[0] != channelsData[channelindex].shape[0]:
    #     # print(datafile)
    #
    #     # print("annotations", annotations[0], annotations[1], annotations[2])
    #     fileh5['label'] = labels
    # saveExcelFile(labels, sc_h5file + "NoLabelData/label/label.xls")
    # print(len(labels), len(channelsData[0]))
    # print("提取数据结束-----")

    # file_num = -1
    #
    # getChannelData(sc_data_path + sc_data_files[file_index], sc_data_path + sc_data_files[file_index + 1],
    #                sc_h5file + "data/" + str(file_num + 1).zfill(4) + ".h5",
    #                sc_h5file + "label/" + str(file_num + 1).zfill(4), 3, 100)
    # # 提取st的数据
    # file_indexs = np.array(range(len(st_data_files)))[::2]
    # for file_index in file_indexs:
    #
    #     getChannelData(st_data_path + st_data_files[file_index], st_data_path + st_data_files[file_index + 1], st_h5file + "data/" + str(file_num + 1).zfill(4) + ".h5",
    #                    st_h5file + "label/" + str(file_num + 1).zfill(4), 3, 100)
    #     file_num += 1







# #转换文件格式
# def loadData(edffile,h5file,labelfile):
#     #对所有文件进行转换
#     for filei in range(5):
#         fileedf = pyedflib.EdfReader(edffile[filei])
#         fileh5 = h5py.File(h5file[filei],'w')
#         signal_headers = fileedf.getSignalHeaders()
#         #print("headers:",signal_headers)
#         #print("参数",signal_headers[filei]['sample_rate'],signal_headers[filei]['digital_max'],signal_headers[filei]['digital_min'])
#         fileh5['digital_max'] = np.array([signal_headers[filei]['digital_max']])
#         fileh5['digital_min'] = np.array([signal_headers[filei]['digital_min']])
#         #print("参数结果",fileh5['sample_rate'].value,fileh5['digital_max'].value,fileh5['digital_min'].value)
#
#         row = 0
#         for signi in list(range(len(signal_headers))):
#             column = 120*30 #一个30秒epoch的列数
#             size = fileedf.readSignal(signi, 0, None, False).size  #获取数据点总数
#             row = size//column ##行数
#             oldsignals=fileedf.readSignal(signi, 0, size, False)
#             newsignals=oldsignals
#             newSignal=len(newsignals)//column * column ##epochs总数
#             fileh5[str(signal_headers[signi]['label'])] = newsignals[0:newSignal].reshape(-1,column)  #存储数据点
#         fileh5['sample_rate'] = np.array([120],dtype=int)
#
#         #将标签加入数据文件中
#         with open(labelfile[filei],"r") as f:
#             csv_read = csv.reader(f)
#             labels = []
#             #print("行数",row)
#             for labelsum,label in enumerate(csv_read):
#                 if labelsum != row :
#                     labels.append(label[0])  #或许对应数量的标签
#                 else:
#                     break
#             labels = np.array(labels,dtype='int32').reshape(-1,1) #转换类型
#             print(labels.shape)
#             fileh5['labels'] = labels
#         #print(fileh5['labels'])
#         #print("---------------")
#         fileedf.close()
# print("开始")
# loadData(edffile,h5file,labelfile)
# print("结束")


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

