# 对于一般地线性回归问题，参数的求解采用是最小二乘法，其目标函数如下
# argmin||Xw-y||²
# 参数w的求解，也可以使用如下矩阵方法进行：
# w=(X^(T)X)^(-1)X^Ty
# 对于矩阵X，若某些列线性相关性较大(即训练样本中某些属性线性相关)
# 就会导致X^(T)X的值接近0，在计算(X^(T)X)^(-1)时就会出现不稳定性
# 传统的基于最小二乘的线性回归法缺乏稳定性
# 岭回归(ridge regression)是一种专用于共线性数据分析的有偏估计回归方法
# 是一种改良的最小二乘估计法，对某些数据的拟合要强于最小二乘法

# 在sklearn中，可以使用sklearn.linear_model.Ridge调用岭回归模型
# 其主要参数有
# alpha：正则化因子，对应于损失函数中的α
# fit_intercept：表示是否计算截距
# solver：设置计算参数的方法，可选参数auto，svd，sag等

# 数据介绍
# 某路口的交通流浪检测数据，记录全年小时级别的车流量
# 实验目的
# 根据已有的数据创建多项式特征，使用岭回归模型代替一般的线性模型，对车流量的信息进行多项式回归

# 1. 建立工程，导入sklearn相关的工具包
import numpy as np  # 导入numpy工具包
from sklearn.linear_model import Ridge  # 通过sklearn.linear_model加载岭回归方法
from sklearn import cross_validation  # 加载交叉验证模块
import matplotlib.pyplot as plt  # 加载matplotilib模块
# 通过sklearn.preprocessing加载PolynomialFeatures，用于创建多项式特征，如ab，a²，b²
from sklearn.preprocessing import PolynomialFeatures

# 2. 数据加载
# 使用numpy方法从cvs文件中加载数据
data = np.genfromtxt('data.csv', delimiter=',', usecols=(1, 2, 3, 4, 5))
# delimiter=','表示按逗号','分隔
# usecols=(1, 2, 3, 4, 5)表示采用数据中的哪几列
# 本例中总共采集了5列数据，数据的第2列(下标为1)到第6列(下标为5)
# 这五列分别代表了数据的5个特征：
# HR：一天中的第几个小时(0-23)
# WEEK_DAY：一周中的第几天(0-6)
# DAY_OF_YEAR：一年中的第几天(1-365)
# WEEK_OF_YEAR：一年中的第几周(1-53)
# TRAFFIC_COUNT：交通流量

# 使用plt展示车流量信息
# plt.plot(data[:, 4]) # 展示车流量数据
# plt.show() # 显示图形
# 注释掉上述两行，从而不影响后面plt的使用

# 3. 数据处理
X = data[:, :4]

# X用于保存0-3维数据，即属性
y = data[:, 4]
# y用来保存第4维数据，即车流量
poly = PolynomialFeatures(6)
# 用来创建最高次数6次方的多项式特征，多次试验后决定采用6次
X = poly.fit_transform(X)
# X为创建的多项式特征

# 4. 划分训练集和测试集
# 将所有数据划分为训练集和测试集
# test_size表示测试集的比例
# random_state是随机数种子
train_set_X, test_set_X, train_set_y, test_set_y = cross_validation.train_test_split(X, y, test_size=0.3,
                                                                                     random_state=0)
# 5. 创建回归器，并进行训练
clf = Ridge(alpha=1.0, fit_intercept=True)
# 调用fit函数使用训练集训练回归器
clf.fit(train_set_X, train_set_y)
# 利用测试集计算回归曲线的拟合优度
# clf.score返回值为0.7375
# 拟合优度，用于评价拟合好坏，最大为1，无最小值
# 当对所有输入都输出同一个值时，拟合优度为0

# 6. 画出拟合曲线
# 画一段200到300范围内的拟合曲线
start = 200
end = 300
# 调用predict函数的拟合值
y_pre = clf.predict(X)
time = np.arange(start, end)
# 展示真实数据(蓝色)以及拟合的曲线(红色)
plt.plot(time, y[start:end], 'b', label='real')
plt.plot(time, y_pre[start:end], 'r', label='predict')
# 设置图例的位置
plt.legend(loc='upper left')
plt.show()
