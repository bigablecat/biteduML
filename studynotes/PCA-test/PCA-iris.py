# 目标：已知鸢尾花数据是4维的
# 共三类样本
# 使用 03PCA-test 实现对鸢尾花 数据进行降维
# 实现在二平面上的可视化。

import matplotlib.pyplot as plt  # 加载matplotlib用于数据可视化
from sklearn.decomposition import PCA  # 加载PCA算法包
from sklearn.datasets import load_iris  # 加载鸢尾花数据集导入函数

data = load_iris()  # 以字典形式加载鸢尾花数据集
y = data.target  # 使用y表示数据集中的标签
X = data.data  # 使用x表示数据集中的属性数据
pca = PCA(n_components=2)  # 加载PCA算法,设置将为后主成分数目为2
reduced_X = pca.fit_transform(X)  # 对原始数据进行降维,保存在reduce_x中

red_x, red_y = [], []  # 第一类数据点
blue_x, blue_y = [], []  # 第二类数据点
green_x, green_y = [], []  # 第三类数据点

for i in range(len(reduced_X)):
    # 注意y是原始数据集data中的标签
    # y[i] == 0 表示当前标签属于第一类样本
    # y[i]=1和y[i]=2以此类推
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        # reduced_X是一个列表
        # reduced_X[i]获取列表的第i个元素
        # reduced_X[i]是一个包含新数据点横纵坐标两个元素的列表??为啥是这样
        # reduced_X[i][0]是横坐标,所以存入red_x
        red_y.append(reduced_X[i][1])
        # reduced_X[i][0]是横坐标,所以存入red_y
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x, red_y, c='r', marker='x')  # 第一类数据点
# matplotlib.pyplot.scatter(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, hold=None, data=None, **kwargs)
# x,y是长度相等的数组,即输入数据
# c定义颜色,c='r',c='b',c='g'分别代表red,blue,green
# marker定义图上散点的标记样式,可用string或者数组类型,默认为None
# marker='x'表示用叉标记
plt.scatter(blue_x, blue_y, c='b', marker='D')  # 第二类数据点
# marker='D'表示用方块标记
plt.scatter(green_x, green_y, c='g', marker='.')  # 第三类数据点
# marker='.'表示用点标记
plt.show()  # 可视化,显示散点图
