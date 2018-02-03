# 决策树是一种树形结构的分类器，通过顺序询问分类点的属性决定最终类别
# 通常根据特征的信息增益或其他指标，构建一棵决策树
# 在分类时，只需要按照决策中的结点依次进行判断，即可得到样本所属类别

# sklearn.tree.DecisionTreeClassifier创建一个决策树用于分类
# 其主要参数有：
# • criterion：用于选择属性的准则，可以传入"gini"代表基尼系数，或者"entropy"代表信息增益
# • max_features：表示在决策树结点进行分裂时，从多少个特征中选择最优特征
# 可以设定固数目、百分比或其他标准,它的默认值是使用所有特征个数

from sklearn.datasets import load_iris  # 导入鸢尾花数据集
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.model_selection import cross_val_score  # 导入计算交叉验证值的函数corss_val_score

# 使用默认参数，创建一棵基于尼系的决策树并将该分类器赋值给变量clf
clf = DecisionTreeClassifier()

# 将鸢尾花数据赋值给变量iris
iris = load_iris()

# 将决策树分类器做为待评估的模型
# iris.data鸢尾花数据做为特征
# iris.target鸢尾花分类标签做为目标结果
# 通过设定cv为10,使用10折交叉验证,得到最终的交叉验证得分
scores = cross_val_score(clf, iris.data, iris.target, cv=10)

print(scores)

# 决策树本质上是寻找一种对特征空间的划分
# 旨在构建训练数据拟合好，并且复杂度小的决策树
# 在实际使用中，需要根据数情况调整DecisionTreeClassifier类中传入的参数
# 比如选择合适的criterion,设置随机变量
