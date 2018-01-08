from numpy.random import RandomState
# 加载RandomState用于创建随机种子
import matplotlib.pyplot as plt
# 加载matplotlib用于数据的可视化
from sklearn.datasets import fetch_olivetti_faces
# 加载Olivetti人脸数据集导入函数
from sklearn import decomposition

# 加载PCA算法包

n_row, n_col = 2, 3
# 设置图像展示时的排列情况
# 此处#n_row表示行,n_col表示列
n_components = n_row * n_col
# 设置提取的特征数目,此处为行列相乘
image_shape = (64, 64)
# 设置人脸数据图片的大小

###############################################################################
# Load faces data
dataset = fetch_olivetti_faces(shuffle=True, random_state=RandomState(0))
# shuffle : boolean, optional # 如果shuffle为True表示打乱数据避免同个人分到一组
# If True the order of the dataset is shuffled to avoid having images of the same person grouped.
# random_state : optional, integer or RandomState object 随机种子用于乱序
# The seed or the random number generator used to shuffle the data.

faces = dataset.data  # faces是获取到的数据


###############################################################################
def plot_gallery(title, images, n_col=n_col, n_row=n_row):
    plt.figure(figsize=(2. * n_col, 2.26 * n_row))  # 创建图片,指定图片大小(英寸)
    plt.suptitle(title, size=16)  # 设置标题及字号大小

    for i, comp in enumerate(images):
        # enumerate python的枚举类型
        # 用于既要遍历索引又要遍历元素的情况
        # 此处i是索引,comp是元素
        plt.subplot(n_row, n_col, i + 1)  # 选择画制的子图
        vmax = max(comp.max(), -comp.min())

        plt.imshow(comp.reshape(image_shape), cmap=plt.cm.gray,
                   interpolation='nearest', vmin=-vmax, vmax=vmax)
        # 对数值归一化,并以灰度图像式显示
        plt.xticks(())  # 去除子图坐标轴标签
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.94, 0.04, 0.)  # 对字图位置及间隔调整


plot_gallery("First centered Olivetti faces", faces[:n_components])
###############################################################################

estimators = [
    ('Eigenfaces - PCA using randomized SVD',  # 方法名称
     decomposition.PCA(n_components=6, whiten=True)),  # 使用PCA方法

    ('Non-negative components - NMF',  # 方法名称
     decomposition.NMF(n_components=6, init='nndsvda', tol=5e-3))  # 使用NMF方法
]

###############################################################################

for name, estimator in estimators:  # 分别调取PCA和NMF
    print("Extracting the top %d %s..." % (n_components, name))
    print(faces.shape)
    estimator.fit(faces)  # 调用PCA或NMF提取特征??
    components_ = estimator.components_  # 获取提取的特征??
    plot_gallery(name, components_[:n_components])  # 按照固定格式进行排列 ??

plt.show()  # 调用show()函数展示
