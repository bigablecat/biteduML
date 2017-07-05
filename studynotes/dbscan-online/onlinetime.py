import numpy as np
import sklearn.cluster as skc
from sklearn import metrics
import matplotlib.pyplot as plt

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

mac2id = dict()  # 创建一个字典变量
onlinetimes = []  # 创建一个列表
f = open('TestData.txt', encoding='utf-8')  # 读取数据
for line in f:  # 读取每行数据
    mac = line.split(',')[2]  # 获取mac地址,以","逗号分隔每行,其中mac地址为第3个元素,用下标2获取
    onlinetime = int(line.split(',')[6])  # 上网时长
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])  # 开始上网时间的整点数
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
