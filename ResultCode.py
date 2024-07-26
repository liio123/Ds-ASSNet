from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from copy import deepcopy
import matplotlib as plt
import os
import h5py
from sklearn import preprocessing
import torch.utils.data as Data
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
from sklearn.model_selection import KFold
import copy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import ast
from sklearn.metrics import precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import label_binarize


def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)


def cm_plot(original_label, predict_label, knum, savepath):
    cm = confusion_matrix(original_label, predict_label)

    cm_new = np.zeros(shape=[5, 5])
    for x in range(5):
        t = cm.sum(axis=1)[x]
        for y in range(5):
            cm_new[x][y] = round(cm[x][y] / t * 100, 2)

    plt.matshow(cm_new, cmap=plt.cm.Blues)

    plt.colorbar()
    x_numbers = []
    y_numbers = []
    cm_percent = []
    for x in range(5):
        y_numbers.append(cm.sum(axis=1)[x])
        x_numbers.append(cm.sum(axis=0)[x])
        for y in range(5):
            percent = format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f")
            cm_percent.append(str(percent))
            plt.annotate(format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f"), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=10)

    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    plt.ylabel('True label')

    y_stage = ["W\n(" + str(y_numbers[0]) + ")", "N1\n(" + str(y_numbers[1]) + ")", "N2\n(" + str(y_numbers[2]) + ")",
               "N3\n(" + str(y_numbers[3]) + ")", "REM\n(" + str(y_numbers[4]) + ")"]
    x_stage = ["W\n(" + str(x_numbers[0]) + ")", "N1\n(" + str(x_numbers[1]) + ")", "N2\n(" + str(x_numbers[2]) + ")",
               "N3\n(" + str(x_numbers[3]) + ")", "REM\n(" + str(x_numbers[4]) + ")"]
    y = [0, 1, 2, 3, 4]
    plt.xticks(y, x_stage)
    plt.yticks(y, y_stage)

    plt.tight_layout()
    plt.savefig("%smatrix%s.svg" % (savepath, str(knum)), bbox_inches='tight')  ##bbox_inches用于解决显示不全的问题
    # plt.show()
    plt.show()
    plt.close()
    return kappa(cm), classification_report(original_label, predict_label, digits=6, output_dict=True)

def cm_plot_number(original_label, predict_label, knum, savepath):
    cm = confusion_matrix(original_label, predict_label)

    cm_new = np.zeros(shape=[5, 5])
    for x in range(5):
        t = cm.sum(axis=1)[x]
        for y in range(5):
            cm_new[x][y] = round(cm[x][y] / t * 100, 2)

    plt.matshow(cm_new, cmap=plt.cm.Blues)

    plt.colorbar()
    x_numbers = []
    y_numbers = []
    for x in range(5):
        y_numbers.append(cm.sum(axis=1)[x])
        x_numbers.append(cm.sum(axis=0)[x])
        for y in range(5):
            percent = format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f")

            plt.annotate(format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f"), xy=(y, x),
                         horizontalalignment='center',
                         verticalalignment='center', fontsize=10)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')

    y_stage = ["W\n(" + str(y_numbers[0]) + ")", "N1\n(" + str(y_numbers[1]) + ")", "N2\n(" + str(y_numbers[2]) + ")",
               "N3\n(" + str(y_numbers[3]) + ")", "REM\n(" + str(y_numbers[4]) + ")"]
    x_stage = ["W\n(" + str(x_numbers[0]) + ")", "N1\n(" + str(x_numbers[1]) + ")", "N2\n(" + str(x_numbers[2]) + ")",
               "N3\n(" + str(x_numbers[3]) + ")", "REM\n(" + str(x_numbers[4]) + ")"]
    y = [0, 1, 2, 3, 4]
    plt.xticks(y, x_stage)
    plt.yticks(y, y_stage)
    # sns.heatmap(cm_percent, fmt='g', cmap="Blues", annot=True, cbar=False, xticklabels=x_stage, yticklabels=y_stage)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒

    plt.tight_layout()
    plt.savefig("%smatrix%s.svg" % (savepath, str(knum)), bbox_inches='tight')  ##bbox_inches用于解决显示不全的问题
    # plt.show()
    plt.show()
    plt.close()
    # plt.savefig("/home/data_new/zhangyongqing/flx/pythoncode/"+str(knum)+"matrix.jpg")
    return kappa(cm), classification_report(original_label, predict_label, digits=6, output_dict=True), cm

##保存结果报告文件
def saveReportFile(file_name, contents):
    fh = open(file_name, 'w')
    fh.write(contents)
    fh.close()

##保存Excel文件，并指定小数位数
def saveExcelFile(file_name, contents):
    # 写
    df = pd.DataFrame(contents).transpose().round(4)
    df.to_csv(file_name)

def save_data(knum, filenum,label,pred):
    test_pred_path = "D:/WorkSpace/Shell/MNN/result_new20/test_pred_files/"  # 数据路径
    filename1 = test_pred_path +str(knum)+ "_test_pred_" + str(filenum) + ".h5"
    f1 = h5py.File(filename1, 'w')
    f1['label'] = label
    f1['result'] = pred
    # print(len(lable),len(x_data))
    f1.close()


#画睡眠特征图
def plot_sleepPicture(label, predict, savepath, filename):
    ##替换为正确的stage
    label_new = replace_stage(copy.deepcopy(label))
    predict_new = replace_stage(copy.deepcopy(predict))
    plot_sleep_label_pred(label_new, predict_new, savepath, filename + "label_pred.svg")  # 画出睡眠结构图
    plot_sleep(label_new, savepath, filename + "label.svg", "red")  # 画出睡眠结构图
    plot_sleep(predict_new, savepath, filename + "pred.svg", "blue")  # 画出睡眠结构图



##将睡眠分期对应的数字替换正确
def replace_stage(old_array):
    old_array[old_array == 4] = 5
    old_array[old_array == 3] = 4
    old_array[old_array == 2] = 3
    old_array[old_array == 1] = 2
    old_array[old_array == 0] = 1
    return old_array

#画出睡眠特征图
def plot_sleep(label, savepath, filename, color):
    x = range(0, len(label))
    y = np.arange(5)
    print(len(label))
    y_stage = ["W", "N1", "N2", "N3", "REM"]
    plt.figure(figsize=(16, 5))
    plt.ylabel("Sleep Stage")
    plt.xlabel("30s Epoch(120 epochs = 1 hour)")
    # plt.xticks(range(0, len(label), int(len(label)//10)), range(0, len(label), int(len(label)//10)))  ##为了将坐标刻度设为字
    plt.yticks(y, y_stage)  ##为了将坐标刻度设为字
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.xlim(0, len(label))
    plt.plot(x, label, linestyle='-', color=color, alpha=1, linewidth=1)
    # plt.plot(x, label, 'r-', color='red', alpha=1, linewidth=1, label='label')
    plt.legend(loc='best')

    plt.savefig("%s/%s" % (savepath, filename))  ##保存睡眠模型图文件
    plt.show()
    plt.close()




#画出睡眠特征图
def plot_sleep_label_pred(label, pred, savepath, filename):
    x = range(0, len(label))
    y = np.arange(5)
    y_stage = ["W", "REM", "N1", "N2", "N3"]
    plt.figure(figsize=(16, 5))
    plt.ylabel("Sleep Stage")
    plt.xlabel("Sleep Time")

    plt.yticks(y, y_stage)  ##为了将坐标刻度设为字
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.xlim(0, len(label))
    plt.plot(x, label, linestyle='-', color='red', alpha=1, linewidth=1, label='label')
    plt.plot(x, pred, linestyle='-', color='blue', alpha=1, linewidth=1, label='predict')

    plt.legend(loc='upper left')
    plt.savefig("%s/%s" % (savepath, filename))   ##保存睡眠模型图文件
    plt.close()
    plt.show()

##PCA  做了之后，svm分类效果变差
def PCA_Featrue(feature_list,num):
    # ##进行PCA分析
    pca = PCA(n_components=num) #n_components代表保存的特征数
    pca.fit(feature_list)
    low_feature = pca.transform(feature_list)   #降低维度
    print(pca.explained_variance_ratio_)  ##贡献值

    return low_feature



def plotLineChart(feature, path):
    x = range(feature.shape[0])
    # line 的设置
    plt.figure(figsize=(16, 8))
    plt.plot(x, feature,
             color='green',  # 线条颜色
             #linestyle='-',  # 线型

             alpha=0.9,  # 透明度
             linewidth=1)  # 线宽
    plt.savefig(path)
    #plt.show()

# TSNE
def divide_data_based_on_labels(datas, labels):

    # print(datas.shape)
    # print(labels.shape)

    index_dict = {}
    datas_dict = {}

    # print(labels)
    for i, num in enumerate(labels):
        # new_num = one_hot_to_number(num)
        num = num.item()
        if num not in index_dict:

            index_dict[int(num)] = [i]
            datas_dict[int(num)] = [datas[i,:]]
            # print("aaa")
        else:
            index_dict[int(num)].append(i)
            datas_dict[int(num)].append(datas[i, :])
            # print("bbb")


    unique_labels = list(index_dict.keys())
    print(unique_labels)
    return datas_dict, unique_labels

def tSNE_2D(datas, labels, label_names, name):
    # datas = datas.cpu().detach().numpy()
    print(datas.shape)
    # labels = labels.cpu()
    # print(type(datas))
    datas = datas.reshape(datas.shape[0], -1)

    # Perform t-SNE dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    datas_tsne = tsne.fit_transform(datas)
    # print(type(datas_tsne))

    # print(datas_tsne.shape)

    datas_tsne, unique_labels = divide_data_based_on_labels(datas_tsne, labels)
    # print(datas_tsne)
    print(np.array(unique_labels).shape)

    # datas_tsne = np.array(datas_tsne[1])
    # print(datas_tsne[:,0])

    markers = ['*', 'o', 'v', 'd', 'x']
    # markers = ['o', 'o', 'o', 'o', 'o']
    # markers = ['.', '.', '.', '.', '.']

    # Plot the data in 2D
    plt.figure(figsize=[15, 10])
    plt.xticks([])
    plt.yticks([])
    index = 0
    for label in unique_labels:
        # print(label)
        datas_array_tsne = np.array(datas_tsne[label])
        # print(label_names)
        plt.scatter(datas_array_tsne[:, 0], datas_array_tsne[:, 1], marker=markers[index])
        index += 1
    # plt.title(title)
    # plt.xlabel('Dimension 1')
    # plt.ylabel('Dimension 2')
    plt.legend(prop={'family' : 'Times New Roman', 'size' : 12}, labels=["Wake", "N1", "N2", "N3", "REM"])
    plt.savefig(f'C:/Users/A/Desktop/ts/TSNE_Uestc_{name}.svg', format='svg', dpi= 120, bbox_inches='tight')
    plt.show()

def draw_PRAUC(labels, pre_lab):
    # 将真实标签二值化
    y_test_binarized = label_binarize(labels, classes=[0, 1, 2, 3, 4])
    # 计算每个类别的精确率和召回率，并绘制P-R曲线
    plt.figure()
    lw = 2
    class_list = ['W', 'N1', 'N2', 'N3', 'REM']
    colors = ['navy', 'magenta', 'darkorange', 'aqua', 'cornflowerblue']
    for i in range(y_test_binarized.shape[1]):
        precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], pre_lab[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=colors[i], lw=lw, label=f'Class {class_list[i]} (area = {pr_auc:.2f})')
    plt.plot([0, 1], [1, 0], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig('C:/Users/A/Desktop/prauc/UestcPRAUC20.svg', format='svg', dpi=120, bbox_inches='tight')
    plt.show()

def ROCAUC(y_test, y_score):
    y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
    n_classes = 5
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):  # 遍历多个类别
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    print('Compute micro-average ROC curve and ROC area')
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Plot FP,TP ROC curves
    lw = 2
    plt.figure()
    # 平均分类 ROC
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average(area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)

    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    colors = ['navy', 'magenta', 'darkorange', 'aqua', 'cornflowerblue']
    classes = ['W', 'N1', 'N2', 'N3', 'REM']

    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='Class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operation Characteristics Curve ')
    plt.legend(loc="lower right")
    plt.savefig("C:/Users/A/Desktop/rocauc/UestcROCAUC10.svg", format='svg', dpi=120, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':


    pass
