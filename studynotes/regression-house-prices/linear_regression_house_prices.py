# 线性回归(Linear Regression) 是利用数理统计中回归分析，来确定两种或以上变量间相互依赖的定关系一统计分析方法
# 线性回归利用称为线性回归方程的最小平函数对一个或多个自变量和因变量之间的关系进行建模
# 这种函数是一个或多称为回归系数的模型参数的线性组合
# 只有一个自变量情况称为简单回归,大于一个自变量情况的叫做多元回归
# 线性回归有很多实际的用途，分为以下两类：
# 1. 如果目标是预测或者映射，线性回归可以用来对观数据集的y和X的值拟合出一个预测模型。
# 当完成这样一个模型以后，对于新增X值，在没有给定与它相配对的y的情况下，可以用这个拟合过的模型预测出一y值
# 2. 给定一个变量y和一些变量X1,⋯,𝑋𝑝,这些变量有可能与y相关
# 线性回归分析可以用来量化y与X𝑗之间相关性的强度，评估出与y不相关的X𝑗，并识别出哪些X𝑗的子集包含了关于y的冗余信息

# 背景：
# 与房价密切相关的除了单位，还有屋尺寸。
# 根据已知的房屋成交价和尺寸进行线性回归，继而可以对已知房屋尺寸而未知房屋成交价格的实例进行成交价格预测
# 目标：
# 对房屋成交信息建立回归方程，并依据回归方程对房屋价格进行预测
# 简单而直观的方式是通过数据可视化直接观察房屋成交价格与屋尺寸间是否存在线性关系
# 对于本实验的数据来说，散点图就可以很好将其在二维平面中进行可视化表示

# 使用算法：线性回归
# 实现步骤：
# 1. 建立 工程并导入 sklearn 包
# 2. 加载 训练 数据，建立回归方程
# 3. 可视化处理
import matplotlib.pyplot as plt  # matplotlib的pyplot子库提供了和matlab类似的绘图api，方便用户快速绘制2D图表
import numpy as np  # numpy是python语言的一个扩充程序库。支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
from sklearn import linear_model  # 导入线性回归模块

# 建立datasets_X和datasets_Y用来存储数据中的房屋尺寸和房屋成交价格
datasets_X = []
datasets_Y = []
# 打开数据集所在文件prices.txt，读取数据
fr = open('prices.txt', 'r')
lines = fr.readlines()  # 读取整个文件
for line in lines:  # 逐行操作，遍历所有数据
    # 去除数据文件中的逗号
    items = line.strip().split(',')
    # 将读取的数据转换为int型，分别写入datasets_X和datasets_Y
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))

length = len(datasets_X)  # 求得datasets_X的长度，即为数据的总数
# 将datasets_X转化为数组，接着用reshape转为二维，以符合线性回归拟合函数输入参数的要求
datasets_X = np.array(datasets_X).reshape([length, 1])
# 将datasets_Y转为数组
datasets_Y = np.array(datasets_Y)

minX = min(datasets_X)
maxX = max(datasets_X)
# 以数据datasets_X的最大值和最小值为范围，建立等差数列，方便后续画图
X = np.arange(minX, maxX).reshape([-1, 1])

# 调用线性回归模块，建立回归方程
linear = linear_model.LinearRegression()
# 调用sklearn.linear_model.LinearRegression()所需参数
# fit_intercept: 布尔型参数，表示是否计算该模型截距，可选参数
# normalize：布尔型参数，若为True则X在回归前进行归一化，可选参数。默认值为False
# copy_X：布尔型参数，若为True，则X将被复制；否则将被覆盖。可选参数，默认值为True
# n_jobs：整型参数，表示用于计算的作业数量，若为-1，则用所有CPU；可选参数，默认值为1

# 拟合函数
linear.fit(datasets_X, datasets_Y)
# 线性回归fit函数用于你和输入输出数据，调用形式为linear.fit(X,y,sample_weight=None):
# X: X为训练向量
# y：y为相对于X的目标向量
# sample_weight：分配给各个样本的权重数组，一般不需要使用，可省略

# 如果有需要，可以通过两个属性查看回归方程的系数与截距
# 具体代码如下：
# 查看回归方程系数：
# print('Coefficients', linear.coef_)
# 查看回归方程截距
# print('intercept',linear.intercept)

# scatter函数用于绘制数据点，这里表示用红色绘制数据点
plt.scatter(datasets_X, datasets_Y, color='red')
# plot函数用于绘制直线，这里表示用蓝色绘制回归线
plt.plot(X, linear.predict(X), color='blue')
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
