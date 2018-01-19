# 目标：已知鸢尾花数据是4维的
# 共三类样本
# 实现对鸢尾花 数据进行降维
# 实现在二平面上的可视化。

import matplotlib.pyplot as plt  # 引入matplotlib.pyplot用于数据可视化
from sklearn.decomposition import PCA  # 引入PCA算法包
from sklearn.datasets import load_iris  # 引入鸢尾花数据集导入函数

data = load_iris()  # 以字典形式加载鸢尾花数据集
# load_iris()返回鸢尾花数据集,返回的数据类型为sklearn.utils.Bunch
# sklearn.utils.Bunch是字典型的数据类型,可以通过key获取相应的值

y = data.target
# load_iris().target是鸢尾花数据集的标签,数据类型是ndarray
# y = data.target得到一个鸢尾花数据标签的ndarray数组

X = data.data
# load_iris().data是鸢尾花数据集中的属性数据,数据类型是ndarray
# X = data.data得到一个鸢尾花数据集属性的ndarray数组

pca = PCA(n_components=2)  # 加载PCA算法,设置降维后主成分数目为2
# sklearn.decomposition.PCA函数

reduced_X = pca.fit_transform(X)  # 对原始数据进行降维,保存在reduce_x中
# reduced_X是得到一个n行两列的多维数组,数据类型ndarray,第一列表示横坐标,第二列表示纵坐标

red_x, red_y = [], []  # 第一类数据点
blue_x, blue_y = [], []  # 第二类数据点
green_x, green_y = [], []  # 第三类数据点

for i in range(len(reduced_X)):
    # 注意y是原始数据集data中的标签
    # y[i] == 0 表示当前标签属于第一类样本
    # y[i]=1和y[i]=2以此类推
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        # 上面讲过reduced_X是一个n行两列的多维数组
        # reduced_X[i]获取列表的第i个元素
        # reduced_X[i]是一个包含新数据点横纵坐标两个元素的列表,类型也是ndarray
        # reduced_X[i][0]获取了reduced_X[i]中下标为0的第一个元素,即坐标轴的横坐标,存入red_x
        red_y.append(reduced_X[i][1])
        # reduced_X[i][1]获取了reduced_X[i]中下标为1的第二个元素,即坐标轴的纵坐标,存入red_y
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
        # 原理同y[i] == 0
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])
        # 原理同y[i] == 0

plt.scatter(red_x, red_y, c='r', marker='x')  # 第一类数据点
# matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None,
# cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None,
# verts=None, edgecolors=None, hold=None, data=None, **kwargs)
# x,y是长度相等的数组,即输入数据
# c定义颜色,c='r',c='b',c='g'分别代表red,blue,green
# marker定义图上散点的标记样式,可用string或者数组类型,默认为None
# marker='x'表示用叉标记
plt.scatter(blue_x, blue_y, c='b', marker='D')  # 第二类数据点
# marker='D'表示用方块标记
plt.scatter(green_x, green_y, c='g', marker='.')  # 第三类数据点
# marker='.'表示用点标记
plt.show()  # 可视化,显示散点图
