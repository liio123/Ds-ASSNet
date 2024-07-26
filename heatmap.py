import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams["font.weight"] = "bold"

a = np.load("D:/PyCharm Community Edition 2020.1.1/sleep/SHNN - TL/result/睡眠数据EEG第二个通道C4-A1结果/all/lstm.npy")
print(a.shape)
print(type(a))
print(a)

#YlGnBu
# ax = sns.heatmap(a, cmap="YlGnBu", annot=False, linewidths=.5)  # 修改颜色，添加线宽
# plt.show()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

sns.set()
plt.rcParams['font.sans-serif']='SimHei'#设置中文显示，必须放在sns.set之后
np.random.seed(0)
uniform_data = np.random.rand(10, 12) #设置二维矩阵
f, ax = plt.subplots(figsize=(9, 6))

#heatmap后第一个参数是显示值,vmin和vmax可设置右侧刻度条的范围,
#参数annot=True表示在对应模块中注释值
# 参数linewidths是控制网格间间隔
#参数cbar是否显示右侧颜色条，默认显示，设置为None时不显示
#参数cmap可调控热图颜色，具体颜色种类参考：https://blog.csdn.net/ztf312/article/details/102474190
sns.heatmap(a, ax=ax,cmap='viridis',annot=False,cbar=True,yticklabels=False)

# ax.set_title('hello') #plt.title('热图'),均可设置图片标题
ax.set_ylabel('Time Point')  #设置纵轴标签
ax.set_xlabel('Sleep Staging')  #设置横轴标签
ax.set_xticklabels(["W","N1","N2","N3","REM"])
#设置坐标字体方向，通过rotation参数可以调节旋转角度
label_y = ax.get_yticklabels()
plt.setp(label_y, rotation=360, horizontalalignment='right')
# label_x = ax.get_xticklabels()
# plt.setp(label_x, rotation=45, horizontalalignment='right')

plt.show()
