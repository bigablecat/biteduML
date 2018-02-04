# 需要从特征文件和标签中将所有数据加载到内存中
# 由于存在缺失值此步骤还需要进行简单的数据预处理
# • 创建对应的分类器，并使用训练数据进行训练
# • 利用测试集预测，通过使用真实值和预测值的比对，计算模型整体的准确率和召回，来评测模型。

import pandas as pd  # 导入numpy库
import numpy as np  # 导入numpy库

from sklearn.preprocessing import Imputer  # 从sklearn库中导入预处理模块 Imputer
from sklearn.cross_validation import train_test_split  # 导入自动生成训练集和测试的模块 train_test_split
from sklearn.metrics import classification_report  # 导入预测结果评估模块 classification_report

from sklearn.neighbors import KNeighborsClassifier  # 导入K近邻分类器KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯函数GaussianNB


# 数据导入函数，设置传入两个参数，分别是
# 特征文件的列表feature_paths和标签文件的列表label_paths
def load_datasets(feature_paths, label_paths):
    # 读取特征文件列表和标签文件列表中的内容，归并后返回
    feature = np.ndarray(shape=(0, 41))
    # 定义feature数组变量，列和特征维度一致为41
    label = np.ndarray(shape=(0, 1))
    # 定义空的标签变量，列数与签维度一致为1
    for file in feature_paths:
        # 使用pandas库的read_table函数读取一个特征文件的内容
        # 其中指定分隔符为逗号、缺失值为问号且不包含表头
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        # 使用Imputer函数，通过设定strategy参数为'mean'
        # 使用平均值补全缺失值
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        # fit()函数用于训练预处理器
        imp.fit(df)
        # transform()函数用于生成预处理结果
        df = imp.transform(df)
        # 将预处理后的数据加入特征集feature
        feature = np.concatenate((feature, df))
    # 依次遍历完所有特征文件
    for file in label_paths:
        # 调用pandas库中的read_table函数读取一个标签文件的内容
        # 指定分隔符为逗号且不包含表头行
        df = pd.read_table(file, header=None)
        # 由于标签文件没有缺失值，所以直接将读取到的新数据加入labe集合
        label = np.concatenate((label, df))
    # 将标签规整为一维向量
    label = np.ravel(label)
    return feature, label


if __name__ == '__main__':
    # 设置数据路径
    # 因原始数据过大，请自行下载本例中需要的数据
    # 本例中数据的使用方式：
    # 打开官网数据集下载地址：https://pan.baidu.com/s/1eR7doh8
    # 文件路径：mooc课程数据/课程数据/分类/dataset.zip
    # 下载dataset.zip文件
    # 将dataset.zip文件解压,得到A.zip, B.zip, C.zip, D.zip, E.zip, XYZ.zip共6个压缩文件
    # 解压缩其中的A.zip, B.zip, C.zip, D.zip, E.zip, 得到五个文件夹A, B, C, D, E
    # 将这5个文件夹放到与当前py文件(classifier_sports_data.py)相同的目录下即可
    # 注：每个文件夹下都有两个与文件夹同名的文件，文件类型分别是FEATURE文件(.feature)和Property List(.label)

    featurePaths = ['A/A.feature', 'B/B.feature', 'C/C.feature', 'D/D.feature', 'E/E.feature']
    labelPaths = ['A/A.label', 'B/B.label', 'C/C.label', 'D/D.label', 'E/E.label']

    # 使用python的分片方法，将数据路径中前4个值作为训练集
    # 并作为参数传入load_dataset()函数中
    # 得到训练集合的特征x_train，训练集的标签y_train
    x_train, y_train = load_datasets(featurePaths[:4], labelPaths[:4])
    # 将最后一个值对应的数据作为测试集，送入load_dataset()函数中
    # 得到测试集合的特征x_test，测试集的标签y_test
    x_test, y_test = load_datasets(featurePaths[4:], labelPaths[4:])
    # 使用全量数据作为测试集，通过设置测试集比例为test_size=0.0，将数据随机打乱，便于后续分类器的初始化和训练
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size=0.0)

    # 使用默认参数创建K近邻分类器，并将训练集x_train和y_train送入fit()函数进行训练
    # 训练后的分类器保存到变量knn中
    print('Start training knn')
    knn = KNeighborsClassifier().fit(x_train, y_train)
    # 使用训练集x_test进行分类器预测,得到分类结果answer_knn
    print('Training done')
    answer_knn = knn.predict(x_test)
    print('Prediction done')

    # 使用默认参数创建决策树分类器dt，并将训练集x_train和y_train送入fit()函数进行训练
    # 训练后的分类器保存到变量dt中
    print('Start training DT')
    dt = DecisionTreeClassifier().fit(x_train, y_train)
    print('Training done')
    # 使用训练集x_test进行分类器预测，得到分类结果answer_dt
    answer_dt = dt.predict(x_test)
    print('Prediction done')

    # 使用默认参数创建高斯贝叶斯分类器，并将训练集x_train和y_train送入fit()函数进行训练
    # 训练后的分类器保存到变量gnb中
    print('Start training Bayes')
    gnb = GaussianNB().fit(x_train, y_train)
    print('Training done')
    # 使用训练集x_test进行分类器预测，得到分类结果answer_gnb
    answer_gnb = gnb.predict(x_test)
    print('Prediction done')

    # 使用classification_report函数分析分类结果
    # 从精确率precision，召回率recall，f1值f1-score和支持度support四个维度进行衡量
    # 分别使用三种分类器进行输出
    print('\n\nThe classification report for knn:')
    print(classification_report(y_test, answer_knn))
    print('\n\nThe classification report for DT:')
    print(classification_report(y_test, answer_dt))
    print('\n\nThe classification report for Bayes:')
    print(classification_report(y_test, answer_gnb))

    # 从准确度的角度衡量，贝叶斯分类器的效果最好
    # 从召回率和f1值的角度衡量，k近邻效果最好
    # 贝叶斯分类器和k近邻的效果好于决策树

    # 注：此程序运行过程大约十几分钟，视机器状况而异
