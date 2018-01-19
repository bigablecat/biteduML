# 本文件对应原始课件\biteduML\raw_materials\06.py

import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = dict()  # 创建一个字典变量
onlinetimes = []  # 创建一个列表
f = open('TestData.txt', encoding='utf-8')  # 使用读取数据
# open()函数打开一个文件
# • 第一个参数是文件的路径,格式是字符串
# • 路径可以是相对路径,也可以是绝对路径
# • 在windows下用\\分割路径,在OSX下用/分割
# • encoding在原函数的参数列表中不是第二个参数，所以传入参数时注明了参数名称encoding=
# • encoding='utf-8'表示以utf-8字符集读取文件
# open()函数返回的结果的类型是_io.TextIOWrapper

for line in f:  # 解析返回结果f,读取每行数据line,line的类型是str字符串
    mac = line.split(',')[2]  # 获取mac地址
    # 使用split方法,按指定的字符对字符串line进行切片,返回一个字符串列表
    # 将该列表按下标排列,举例如下
    # [0]2c929293466b97a6014754607e457d68,
    # [1]U201215025,
    # [2]A417314EEA7B,
    # [3]10.12.49.26,
    # [4]2014-07-20 22:44:18.540000000,
    # [5]2014-07-20 23:10:16.540000000,
    # [6]1558,
    # [7]15,
    # [8]本科生动态IP模版,
    # [9]100元每半年,
    # [10]internet
    # 其中mac地址为该字符串列表的第3个元素,用下标2获取,举例如A417314EEA7B

    onlinetime = int(line.split(',')[6])  # 上网时长onlinetime是该列表的第5个元素,用下标6获取,加int()转换为整数

    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])  # 开始上网时间的整点数
    # starttime的获取经过了3次切片
    # • 第一次切片：line.split(',')[4]
    # 对整行按逗号切片后,得到第一个列表,上网时间是该列表的第5个元素,用下标[4]获取
    # 获取到的上网时间是一个字符串,格式如 2014-07-20 22:44:18.540000000
    # • 第二次切片：line.split(',')[4].split(' ')[1]
    # 用' '空格对上网时间字符串进行切片，再次得到一个字符串列表
    # 这个新列表的第一个元素是开始上网时间的年月日(例如2014-07-20)
    # 新列表的第二个元素是开始上网时间的时分秒(例如22:44:18.540000000)
    # 用下标[1]获取第二个元素
    # • 第三次切片：line.split(',')[4].split(' ')[1].split(':')[0]
    # 对上网时间的时分秒按冒号切片,获得一个包含时、分、秒三个字符串元素的列表
    # 用下标[0]获取小时的字符串
    # 最终结果用int()转换为整数

    if mac not in mac2id:
        mac2id[mac] = len(onlinetimes)
        # 字典变量mac2id的key为mac地址,value为列表onlinetimes的长度
        # 相当于用mac2id[mac]记录onlinetimes的最大索引
        onlinetimes.append((starttime, onlinetime))
        # (starttime, onlinetime)是一个元组类型的对象，存储最新获取的starttime和onlinetime两个元素
        # 向列表onlinetimes添加当前元组对象
    else:
        # onlinetimes[mac2id[mac]] = [(starttime, onlinetime)] # 课件中的原始代码
        onlinetimes[mac2id[mac]] = (starttime, onlinetime)  # 修正后的代码
        # (starttime, onlinetime)同上,仍然生成一个元组，存储最新获取的starttime和onlinetime两个元素
        # 上面讲到,mac2id[mac]实际是列表onlinetimes的一个索引值
        # onlinetimes[mac2id[mac]]获取onlinetimes中索引值为[mac2id[mac]]的一个元素
        # 用最新的元组(starttime, onlinetime)替换原有元素
        # 原始代码勘误:
        # if中用append添加了一个元组,else中用等号赋值,等号右边应该也是同样格式的元组
        # 所以去除原始代码中的中括号才是正确代码
    # 这组if else语句的作用是:
    # 通过mac2id[mac]中mac地址的唯一性，将onlinetimes索引赋值给mac2id[mac]

real_X = np.array(onlinetimes).reshape((-1, 2))
# numpy.array() 是numpy的一个函数
# numpy.array() 接受一切序列型的对象: 列表,元组,数组等
# numpy.array() 最终生成一个numpy数组(类型是ndarray)
# np.array(onlinetimes)通过传入列表onlinetimes, 生成一个ndarray多维数组
# reshape((-1, 2)) 又对np.array(onlinetimes)生成的这个多维数组进行了reshape操作
# reshape() 函数不改变原数组的元素，只返回一个shape形状的数组
# reshape() 传入的shape参数类型是元组，元组中的每个数字代表一个维度的大小
# 其中的任一维度都可以是数字-1，表示该维度的大小由数据本身推断而来
# np.array(onlinetimes).reshape((-1, 2)),传入的参数是元组(-1, 2)
# 表示将np.array(onlinetimes)获取的数组重塑为n行2列的多维数组

X = real_X[:, 0:1]
# 对多维数组real_X切片,返回一个切片后的多维数组X
# 下标从列表中获得1个值, 切片从列表中取得多个值
# 切片有两个用冒号:分隔的整数,例如sample_list=[1,2,3,4,5], sample_list[1:4]
# 在切片中,第一个整数是切片开始处的下标
# 第二个整数是切片结束处的下标,但是不包括这个下标获取的值
# 例如sample_list=[1,2,3,4,5]中, sample_list[1:4]获取了列表中下标为[1],[2],[3]的值
# 下标包括了冒号左边的整数1,但是不包括冒号右边的整数4
# sample_list[1:4]= [2, 3, 4]
# 切片还可以省略冒号左右两个数字中的任一个,或者两个数字都省略
# 省略冒号左边的数字表示下标取[0],即分片始于列表的开始位置
# 省略冒号右边的数字表示下标取列表的长度,即分片直至列表结尾
# 例如sample_list=[1,2,3,4,5]中, sample_list[:] = [1,2,3,4,5]
# sample_list[:]相当于sample_list[0:5]
# 为什么省略右边的数字可以取到最后一个元素？
# 上面说过，省略右边数字相当于下标取列表的长度,sample_list=[1,2,3,4,5]中列表长度为5，所以取到下标[5]
# 而下标是从[0]开始的，所以下标[4]已经取到了最后一个元素
# 切片的规则是不包括以冒号右边整数为下标获取的值
# 所以sample_list[:]和sample_list[0:5]都能获取整个sample_list列表

# 本段代码中，real_X[:, 0:1]实际上对real_X进行了两次切片操作
# [:, 0:1]中逗号前的切片:表示对real_X列表按照[:]截取
# real_X中的每个元素本身又是一个列表
# [:, 0:1]中逗号后的切片0:1表示对real_X中的每个元素按照[0:1]截取
# X = real_X[:, 0:1] 获取的X是n行1列的多维数组,X中的每个元素都是数组

db = skc.DBSCAN(eps=0.01, min_samples=20).fit(X)

# skc.DBSCAN(eps=0.01, min_samples=20) 本行代码经过了如下计算流程：
# • 以每个点为中心，计算在这个中心的半径eps=0.01范围内，共有多少个点(包括中心点自己),即邻域内总共有多少个点
# • 如果邻域内点的个数超过min_samples=20，这个点是核心点
# • 查看剩余点是否在核心点的邻域内，若在则为边界点，否则为噪声点
# skc.DBSCAN(eps=0.01, min_samples=20)返回的对象类型是 sklearn.cluster.dbscan_.DBSCAN
# fix()方法:
# skc.DBSCAN(eps=0.01, min_samples=20)定义了DBSCAN的基本参数后，使用fix()方法,传入数据集,得到最终结果
# db = skc.DBSCAN(eps=0.01, min_samples=20).fix(X) 返回的对象类型也是 sklearn.cluster.dbscan_.DBSCAN

# DBSCAN(Density-Based Spatial Clustering of Application with Noise)
# DBSCAN算法是一种基于密度的聚类：
# • 聚类的时候不需要预先指定簇个数
# • 最终的簇个数不定

# DBSCAN算法将数据点分为三类：
# • 核心点：在半径eps内含有超过min_samples数目的点
# • 边界点：在半径eps内点的数量小于min_samples，但是落在核心点的邻域内
# • 噪音点：既不是核心也不是边界点

# sklearn.cluster.DBSCAN函数的完整参数如下
# class sklearn.cluster.DBSCAN (eps=0.5, min_samples=5, metric='euclidean',
# metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=1)
# 这里只介绍前3个常用参数,其他参数可以查看sklearn.cluster.DBSCAN的官方文档
# • eps：两个样本被看作近邻的最大距离，超出这个距离不算近邻
# • min_samples：近邻的最少样本数，达到这个数量某个点才能被认为是核心点(包括这个点自身)
# • metric='euclidean': 距离计算方式，DBSCAN使用默认的欧式距离可以满足大部分需求

# eps
# • eps默认值是0.5
# • eps过大，则更多的点会落在核心点的邻域内, 此时类别数可能会减少，本来不应该是一类的样本也会被划为一类
# • 反之eps过小则类别数可能会增大，本来是一类的样本却被划分开

# min_samples
# • min_samples默认值是5
# • 在eps一定的情况下
# • min_samples过大，则核心点会过少, 此时簇内部分本来是一类的样本可能会被标为噪音点，类别数也会变多
# • 反之min_samples过小，则会产生大量的核心点，可能会导致类别数过少

# eps和min_samples常在一起调参，两个值的组合最终影响了聚类效果

# metric常用的度量参数有:
# a) 欧式距离 'euclidean'
# b) 曼哈顿距离 'manhattan'
# c) 切比雪夫距离'chebyshev'
# d) 闵可夫斯基距离'minkowski', 欧氏距离和曼哈顿距离都源自闵可夫斯基距离
# e) 带权重闵可夫斯基距离'wminkowski'
# f) 标准化欧式距离'seuclidean', 即对于各特征维度做了归一化以后的欧式距离, 此时各样本特征维度的均值为0，方差为1.
# g) 马氏距离'mahalanobis'

labels = db.labels_
# labels_是sklearn.cluster.dbscan_.DBSCAN的属性
# 通过sklearn.cluster.dbscan_.DBSCAN的对象db获取.labels_
# labels = db.labels_就是skc.DBSCAN(eps=0.01, min_samples=20).fit(X)计算生成的簇的标签,噪声点的标签是-1
# labels的类型是numpy.ndarray

print('Labels:')
print(labels)  # 将labels这个ndarray打印出来

raito = len(labels[labels[:] == -1]) / len(labels)  # 获取噪声数据占所有数据的比例
# labels[:]用切片方式返回一个包含labels所有元素的新ndarray
# labels[:] == -1 对labels[:]这个ndarray中的所有元素逐个判断是否等于-1, 每次判断都返回一个bool类型的值
# labels[:] == -1 返回一个由bool值组成的ndarray, 这个ndarray的结构和labels[:]原数组完全一致
# labels[labels[:] == -1] 从labels中获取所有bool值为True对应的数值(就是-1), 最终结果是一个ndarray
# 所以labels[labels[:] == -1] 最终返回了一个由数值-1组成的ndarray
# len(labels[labels[:] == -1]) 获得这个-1组成的ndarray的长度,也就是噪声数据的数目
# len(labels)是labels总长度,即labels总数
# raito = len(labels[labels[:] == -1]) / len(labels) 得到噪声点与总数之比

print('Noise raito:', format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# 通过set(labels)对簇标签labels去重
# len(set(labels))得到去重后簇标签的总数,即簇的总数(包括噪声点在内)
# 1 if -1 in labels else 0 遍历labels，判断labels中是否有-1标签，即噪声点
# 如果labels存在噪声点,经过set(labels)去重后,在len(set(labels))就只包含了1个-1
# 用len(set(labels))减1,排除噪声点,得到没有噪声点的簇总数
# 如果不存在噪声点,(1 if -1 in labels else 0得到数值0)
# len(set(labels)) - 0 保持不变

print('Estimated number of clusters: %d' % n_clusters_)
# 打印簇总数n_clusters_

print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))
# sklearn.metrics.silhouette_score 用于计算所有样本轮廓系数的均值
# metrics.silhouette_score(X, labels)
# 参数X是数据集
# 参数labels是数据集的所有簇标签
# metrics.silhouette_score的返回结果是轮廓系数的均值,数据类型是float
# 这个轮廓系数均值用来评价聚类效果

for i in range(n_clusters_):
    print('Cluster ', i, ':')
    print(list(X[labels == i].flatten()))

# python3中的rang()是一个迭代器
# rang()的参数必须是整数类型
# rang()传入一个整数参数时,用来产生从零算起的一系列整数,但不包括这个整数
# rang()传入两个整数参数时,第一个参数视为下边界,即不再从零开始,从第一个参数开始,到第二个参数截止
# rang()传入三个整数参数时,第三个整数作为步进值,函数默认的步进值是1
# range()常用在for循环中产生索引
# range()也常和list()组合产生整数列表
# range(n_clusters_)使用簇的总数为for循环生成了默认步长为1的一系列整数索引
# for i in range(n_clusters_):
#     print('Cluster ', i, ':')
# for循环在python中是一个通用的序列迭代器,可以遍历任何有序的序列对象内的元素
# for循环可拥有字符串、列表、元组、其他内置可迭代对象等
# for i in range(n_clusters_) 本行代码就使用for循环遍历了内置的迭代器rang()函数
# i是从range(n_clusters_)中依次取到的每一个整数
# print('Cluster ', i, ':') 有多少个簇,就打印多少个i
# labels == i 依次检查labels中的每个元素是否与i相等
# labels == i 返回一个结构与labels相同的以bool值组成的新数组
# X[labels == i] 依次检查X中的每个元素,将对应[labels == i]中True的元素取出
# X[labels == i] 返回一个符合条件labels == i的新数组

plt.hist(X, 24)  # 绘制直方图

# matplotlib.pyplot 是一个命令行式风格的函数库,能够实现MATLAB类似的绘图功能
# matplotlib.pyplot.hist 用于绘制直方图
# matplotlib.pyplot.hist 函数有一个很长的参数列表, 这里只讲解前两个参数
# matplotlib.pyplot.hist(x, bins=None)
# 第一个参数x是原始数据，类型是数组或者序列
# 第二个参数bins表示直方图横直方条的个数
# plt.hist(X, 24) 本行代码传入前面获得的多维数组X,按照24小时时段划分为24个直方条

plt.show()  # 课件的源代码中没有这一行,加上这行代码在运行完成后显示matplotlib.pyplot画出的图形

# 参考书目和文章：
# • 《Python编程快速上手——让繁琐工作自动化》,作者Al Sweigart, 译者王海鹏
# • 《Python学习手册》,作者Mark Lutz, 译者李军、刘红伟等
# • 《利用Python进行数据分析》,作者Wes McKinney,译者唐学韬
# • 《用scikit-learn学习DBSCAN聚类》(https://www.cnblogs.com/pinard/p/6217852.html)
# • 《DBSCAN - 基于密度的聚类算法》(http://blog.csdn.net/sandyzhs/article/details/46773731)
# • sklearn.cluster.DBSCAN官方文档(http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
# • 注释中的一些描述，直接从书中或文章中摘取了原句
