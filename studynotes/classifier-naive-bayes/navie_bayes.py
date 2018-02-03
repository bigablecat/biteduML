# 朴素贝叶斯分类器是一个以贝叶斯定理为基础的多分类的分类器
# 对于给定数据，首先基于特征的条件独立性假设
# 学习输入出的联合概率分布
# 然后基于此模型，对给定的输入x，利用贝叶斯定理求出后验概率最大的输出y

# 在sklearnsklearn库中，实现了三个朴素贝叶斯分类器
# naive_bayes.GussianNB高斯朴素贝叶斯
# naive_bayes.MultinomiaNB 针对多项式模型的朴素贝叶斯分类器
# naive_bayes.BernoulliNB 针对多元伯努利模型的朴素贝叶斯分类器

# naive_bayes.GussianNB参数有
# priors：给定各个类别的先验概率
# 如果为空，则按训练数据实际情况进行统计
# 如果给定先验概率,则在训练过程中不能更改

import numpy as np

# 导入numpy库,构造训练数据X和y
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

# 使用import语句导入高斯朴素贝叶斯分类器
from sklearn.naive_bayes import GaussianNB

# 使用默认参数，创建一个高斯朴素贝叶分类器，并将该分类器赋给变量clf
clf = GaussianNB(priors=None)

# 类似的，使用fit()函数进行训练，并使用predict()函数进行预测,得到预测结果为1
# （测试时可以构造二维数组达到同预多个样本的目）
clf.fit(X, Y)
print(clf.predict([[-0.8, -1]]))

# 朴素贝叶斯是典型的生成学习方法，由训练数据联合概率分布并求得后验概率分布
# 朴素 贝叶斯一般在小规模数据上的表现很好，适合进行多分类任务
