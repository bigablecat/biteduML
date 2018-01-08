import numpy as np  # 导入numpy包
from sklearn.cluster import KMeans  # 导入KMeans包
import os


# 定义一个读取数据的方法loadData(filePath),传入参数filePah为数据源文件在计算机中的路径
def loadData(filePath):
    fr = open(filePath, 'r+')  # 打开指定文件,获取原始数据

    # open()函数打开一个文件
    # 第一个参数是文件的路径,格式是字符串
    # 路径可以是相对路径,也可以是绝对路径
    # 在windows下用\\分割路径,在OSX下用/分割
    # 第二个参数是打开方式,默认为'r',表示只读
    # 此处的'r+'表示可读可写 不会创建不存在的文件 从顶部开始写 会覆盖之前此位置的内容
    # open()函数返回一个File对象
    # File对象代表计算机中的一个文件,只是Python中另一种类型的值,类似列表和字典
    # 此处open()函数返回的File文件保存在变量fr里
    # 当需要读写文件时,调用fr变量中File对象的方法

    lines = fr.readlines()  # 读取文件,得到文件中的每一行
    # readline()方法从当前File文件对象取得一个字符串列表
    # 列表中的每个字符串就是文本中的一行

    retData = []  # 定义一个空列表,存储城市的房租数据
    retCityName = []  # 定义一个空列表,存储城市名称数据
    for line in lines:  # 用for循环依次读取数据的每一行,每一行是一个城市
        items = line.strip().split(",")  # 切片后得到一个新的列表items

        # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）
        # Python split() 通过指定分隔符对字符串进行切片
        # str.split(str="", num=string.count(str)). 可传入两个参数str和num
        # str -- 分隔符，默认为所有的空字符，包括空格、换行(\n)、制表符(\t)等。
        # num -- 分割次数。如果参数num 有指定值，则仅分隔 num 个子字符串

        retCityName.append(items[0])  # items中的第一个元素是城市名称,将其加入城市名称列表
        # append() 方法用于在列表末尾添加新的对象。

        retData.append([float(items[i]) for i in range(1, len(items))])

        # 使用列表解析的方法从items的第二个元素开始读取,边读取边将数据转化为float类型
        # range(start, end, scan) 生成一个整数列表
        # 第一个参数start表示起始数字,end表示结尾数字,scan表示步长,默认为1

    return retData, retCityName  # 返回城市名称和城市房租数据两个列表


if __name__ == '__main__':
    #  一个python的文件有两种使用的方法
    # 第一是直接作为脚本执行
    # 第二是import到其他的python脚本中被调用（模块重用）执行
    # 因此if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程
    # 在if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行
    # 而import到其他脚本中是不会被执行的
    data, cityName = loadData('city.txt')  # 调用loadData方法,获取两个列表
    # 'city.txt'使用了相对路径
    km = KMeans(n_clusters=4)  # n_clusters用于指定聚类中心的个数
    # KMeans(n_clusters, init, max_iter) 方法来自sklearn.cluster下的KMeans包
    # n_cluster用于指定聚类中心的个数,一般只用这一个参数
    # init初始聚类中心的初始化方法.默认是k-means++
    # max_iter最大的迭代次数,默认是300
    # 本例中变量km是KMeans返回的一个KMeans类对象
    label = km.fit_predict(data)  # fit_predict计算簇中心，同时为簇分配序号
    # fit_predict 实际上调用了KMeans类中的fit方法，并返回label
    # label是聚类后各数据所属标签
    # Compute cluster centers and predict cluster index for each sample
    expenses = np.sum(km.cluster_centers_, axis=1)
    # numpy.sum(a, axis=None)是numpy的函数
    # 按照给定的轴(第二个参数axis)计算数组(第一个参数a)的和
    # axis默认为None,简单对所有数值相加; axis=0表示按列相加; axis=1表示按行相加
    # 最终返回结果的类型是ndarray
    CityCluster = [[], [], [], []]  # 定义有4个元素的列表
    # 4与之前指定聚类中心的个数相同
    # 每个元素本身又是一个空列表,稍后用来存储城市名称
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])  # 将城市按 label 分成设定的簇
        # label[i]得到一个整数,作为CityCluster的下标
        # CityCluster[label[i]]得到CityCluster中的第label[i]个元素
        # 因为CityCluster中的第label[i]个元素也是列表,所以使用append方法添加元素
        # CityCluster[label[i]].append(cityName[i])就是向列表CityCluster[label[i]]中添加cityName[i]
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])  # 将每个簇的平均花费输出
        # "Expenses:%.2f"中的%.2f是python的格式符，表示将浮点数float保留两位小数
        # "Expenses:%.2f" % expenses[i]完整意思:
        # 将引号外百分号%后的expenses[i]保留两位小数后填入引号内%.2f的位置
        print(CityCluster[i])  # 将每个簇的城市输出
