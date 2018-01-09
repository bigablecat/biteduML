import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

mac2id = dict()  # 创建一个字典变量
onlinetimes = []  # 创建一个列表
f = open('TestData.txt', encoding='utf-8')  # 使用读取数据
# open()函数打开一个文件
# 第一个参数是文件的路径,格式是字符串
# 路径可以是相对路径,也可以是绝对路径
# 在windows下用\\分割路径,在OSX下用/分割
# encoding在原函数的参数列表中不是第二个参数，所以传入参数时注明了参数名称encoding=
# encoding='utf-8'表示以utf-8字符集读取文件
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
    # 其中mac地址为该字符串列表的第3个元素,用下标2获取

    onlinetime = int(line.split(',')[6])  # 上网时长onlinetime是该列表的第5个元素,用下标6获取,加int()转换为整数

    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])  # 开始上网时间的整点数
    # starttime的获取经过了3次切片
    # 第一次切片：line.split(',')[4]
    # 对整行按逗号切片后,得到第一个列表,上网时间是该列表的第5个元素,用下标[4]获取
    # 获取到的上网时间是一个字符串,格式如 2014-07-20 22:44:18.540000000
    # 第二次切片：line.split(',')[4].split(' ')[1]
    # 用' '空格对上网时间字符串进行切片，再次得到一个字符串列表
    # 这个新列表的第一个元素是开始上网时间的年月日(例如2014-07-20)
    # 新列表的第二个元素是开始上网时间的时分秒(例如22:44:18.540000000)
    # 用下标[1]获取第二个元素
    # 第三次切片：line.split(',')[4].split(' ')[1].split(':')[0]
    # 对上网时间的时分秒按冒号切片,获得一个包含时、分、秒三个字符串元素的列表
    # 用下标[0]获取小时的字符串
    # 最终结果用int()转换为整数

    if mac not in mac2id:  #
        mac2id[mac] = len(onlinetimes)
        onlinetimes.append((starttime, onlinetime))
    else:
        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]
real_X = np.array(onlinetimes).reshape((-1, 2))
# np.array(onlinetimes) 通过列表onlinetimes生成一个ndarray多维数组
# reshape() 不改变数组元素，返回一个shape形状的数组，原数组不变
# reshape((-1, 2))

X = real_X[:, 0:1]

db = skc.DBSCAN(eps=0.01, min_samples=20).fit(X)
labels = db.labels_

print('Labels:')
print(labels)
raito = len(labels[labels[:] == -1]) / len(labels)
print('Noise raito:', format(raito, '.2%'))

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

for i in range(n_clusters_):
    print('Cluster ', i, ':')
    print(list(X[labels == i].flatten()))

plt.hist(X, 24)
