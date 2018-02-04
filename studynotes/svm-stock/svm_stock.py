# 网易财经获取的20年上证指数历史数据
# 根据当前时间前150天的历史数据，预测当天上证指数的涨跌
import pandas as pd  # 引入pandas库，用来加载CSV数据
import numpy as np  # 引入numpy库，支持高级大量的维度数组与矩阵运算，此外也针对数组运算提供大量的数学函数库
from sklearn import svm  # svm库
from sklearn import cross_validation  # 交叉验证

data = pd.read_csv('000777.csv', encoding='gbk', parse_dates=[0], index_col=0)
# pandas.read_csv(数据源,encoding=编码格式为gbk,parse_dates=第0列解析为日期 列解析为日期, index_col=用作行索引的列编号)
data.sort_index(0, ascending=True, inplace=True)
# sort_index(axis=0(按0列排 ), ascending=True(升序), inplace=False(排序后是否覆盖原原数据))
# data按升序排列

# 选取150天的数据
dayfeature = 150
# 选列5列数据作为特征：收盘价 最高价 最低价 开盘价 成交量
# 选取5个特征*天数
featurenum = 5 * dayfeature
# 记录150天的5个特征值
x = np.zeros((data.shape[0] - dayfeature, featurenum + 1))

# data.shape[0]-dayfeature意思是因为我们要用150天数据做训练
# 对于条目为200条的数据，只有50条数据是有前150天的数据来训练的
# 所以训练集的大小就是200-150 = 50
# 对于每一条数据，它的特征是前150天的所有特征数据，即150*5
# +1是将当天的开盘价引入作一条特征数据

# 记录涨或者跌
y = np.zeros((data.shape[0] - dayfeature))

for i in range(0, data.shape[0] - dayfeature):
    # u'开盘价'中的u表示unicode编码
    x[i, 0:featurenum] = np.array(data[i:i + dayfeature] \
                                      [[u'收盘价', u'最高价', u'最低价', u'开盘价', u'成交量']]).reshape((1, featurenum))
    # 将数据中的各种价格存入x数组中
    # data.ix中的ix表示索引
    x[i, featurenum] = data.ix[i + dayfeature][u'开盘价']
    # 最后一列记录当日的开盘价

for i in range(0, data.shape[0] - dayfeature):
    # 如果当天收盘价高于开盘价，y[i]=1代表涨,y[i]=0代表跌
    if data.ix[i + dayfeature][u'收盘价'] >= data.ix[i + dayfeature][u'开盘价']:
        y[i] = 1
    else:
        y[i] = 0
# 调用svm函数，并设置kernel参数，默认是rbf，其他参数有linear，poly，sigmoid
clf = svm.SVC(kernel='rbf')
result = []
for i in range(5):
    # x和y的验证集和测试集，切分80-20%的测试集
    x_train, x_test, y_train, y_test = \
        cross_validation.train_test_split(x, y, test_size=0.2)
    # 使用训练数据训练
    clf.fit(x_train, y_train)
    # 将预测数据和测试集的验证数据比对
    result.append(np.mean(y_test == clf.predict(x_test)))
print("svm classifier accuacy:")
print(result)

# 交叉验证法先将数据集D划分为k个大小相似的互斥子集
# 每个子集都尽可能保持数据分布的一致性，即从D中通过分层采样得到
# 然后，每次用k-1个子集的并集作为训练集，余下的那个子集作为测试集
# 这样就可获得k组训练/测试集，从而可进行k次训练和测试，最终返回的是这k个测试结果的均值
# 通常把交叉验证法称为"k折交叉验证",k最常用的取值是10，此时称为10折交叉验证
