# 多项式回归(Polynomial Regression)是研究一个因变量与一个或多个自变量间项式的回归分析方法
# 如果自变量只有一个时，称为一元多项式回归；如果自变量有多个时，称为多元多项式回归
# 在一元回归分析中，如果因变量y与自变量x的关系为非线性的，但是有找不到合适的函数曲线来你和，则可以采用一元多项式回归
# 多项式回归最大的优点就是可以通过增加x的高次项对实测点进行逼近，直至满意为止
# 事实上多项式回归可以处理相当一类非线性问题，它在回归分析中占有重要的地位，因为任一函数都可以分段用多项式逼近

# 之前提到的线性回归事例中，是运用直线来你和数据输入与输出之间的线性关系
# 不同于线性回归，多项式回归是使用曲线你和数据的输入与输出的映射关系

# 背景：
# 与房价密切相关的除了单位，还有屋尺寸。
# 根据已知的房屋成交价和尺寸进行线性回归，继而可以对已知房屋尺寸而未知房屋成交价格的实例进行成交价格预测
# 目标：
# 对房屋成交信息建立多项式回归方程，并依据回归方程对房屋价格进行预测

# 使用算法：多项式回归-线性回归
# 实现步骤：
# 1. 建立 工程并导入 sklearn 包
# 2. 加载 训练 数据，建立回归方程
# 3. 可视化处理
import matplotlib.pyplot as plt  # matplotlib的pyplot子库提供了和matlab类似的绘图api，方便用户快速绘制2D图表
import numpy as np  # numpy是python语言的一个扩充程序库。支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
from sklearn import linear_model  # 导入线性回归模块
from sklearn.preprocessing import PolynomialFeatures  # 导入多项式构造模块

# sklearn中多项式回归
# 这里的多项式回归实际上是先将变量X处理成多项式特征，然后使用线性模型学习多项特征的参数，以达到多项式回归的目的
# 如: 使用PolynomialFeatures构造X的二次多项式特征X_Ploy
# 使用linear_model学习X_Poly和y之间的映射关系

# 建立datasets_X和datasets_Y用来存储数据中的房屋尺寸和房屋成交价格
datasets_X = []
datasets_Y = []
fr = open('prices.txt', 'r')  # 打开数据集所在文件prices.txt，读取数据
lines = fr.readlines()  # 读取整个文件
for line in lines:  # 逐行操作，遍历所有数据
    items = line.strip().split(',')  # 去除数据文件中的逗号
    datasets_X.append(int(items[0]))
    datasets_Y.append(int(items[1]))  # 将读取的数据转换为int型，分别写入datasets_X和datasets_Y

length = len(datasets_X)  # 求得datasets_X的长度，即为数据的总数
# 将datasets_X转化为数组，接着用reshape转为二维，以符合线性回归拟合函数输入参数的要求
datasets_X = np.array(datasets_X).reshape([length, 1])
# 将datasets_Y转为数组
datasets_Y = np.array(datasets_Y)

minX = min(datasets_X)
maxX = max(datasets_X)
# 以数据datasets_X的最大值和最小值为范围，建立等差数列，方便后续画图
X = np.arange(minX, maxX).reshape([-1, 1])

# degree=2表示建立datasets_x的二次多项式特征x_poly
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(datasets_X)

# 然后创建线性回归，使用线性模型学习x_poly和datasets_Y之间的映射关系
lin_reg_2 = linear_model.LinearRegression()
# 调用sklearn.linear_model.LinearRegression()所需参数
# fit_intercept: 布尔型参数，表示是否计算该模型截距，可选参数
# normalize：布尔型参数，若为True则X在回归前进行归一化，可选参数。默认值为False
# copy_X：布尔型参数，若为True，则X将被复制；否则将被覆盖。可选参数，默认值为True
# n_jobs：整型参数，表示用于计算的作业数量，若为-1，则用所有CPU；可选参数，默认值为1
# 拟合函数
lin_reg_2.fit(X_poly, datasets_Y)
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
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
# xlabel和ylabel用来指定横纵坐标的名称
plt.xlabel('Area')
plt.ylabel('Price')
plt.show()
