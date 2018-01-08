# 利用图像的灰度,颜色,纹理,形状等特征,把图像分成若干互不重叠的区域
# 并使这些特征在同一区域内呈现相似性,在不同区域之间存在明显的差异性
# 然后就可以将分割的图像中具有独特性质的区域提取出来用于不同的研究
# 图像分割的应用: 轮毂裂纹图像的分割; 肝脏CT图像的分割
# 图像分割常用方法:
# 1. 阈值分割: 对图像灰度值进行度量,设置不同的阈值,达到分割的目的
# 2. 边缘分割: 对图像边缘进行检测,即检测图像中灰度值发生跳变的地方,则为一片区域的边缘
# 3. 直方图法: 对图像的颜色建立直方图,直方图的波峰波谷能够表示一块区域的颜色值范围,来达到分割的目的
# 4. 特定理论: 基于聚类分析,小波变换等理论完成图像分割
# 本例描述
# 目标: 利用K-means聚类算法对图像像素点进行聚类实现简单的图像分割
# 输出: 同一聚类中的点使用相同的颜色标记,不同聚类颜色不同
# 实验步骤:
# 1. 建立工程并导入sklearn包
# 2. 加载图片进行预处理
# 3. 加载K-means聚类算法
# 4. 对像素点进行聚类并输出
import numpy as np
import PIL.Image as image  # 导入PIL包,用于加载创建图片
from sklearn.cluster import KMeans  # 加载KMeans算法


def loadData(filePath):
    f = open(filePath, 'rb')  # 以二进制的形式打开文件
    # Python 的 open() 下, ‘r’代表可读, 包括'+'代表可读可写
    # 'b'代表二进制模式访问
    # 对于所有POSIX兼容的Unix系统(包括Linux)来说,'b'是可由可无的
    # 因为它们把所有的文件当作二进制文件，包括文本文件
    data = []  # 创建一个名为data的数组
    img = image.open(f)  # PIL.Image.open()方法以列表形式返回图像像素值
    m, n = img.size  # 获得图片的大小,纵值m,横值n
    for i in range(m):
        for j in range(n):  # 将每个像素点RGB颜色处理到0-1范围内并放进data
            x, y, z = img.getpixel((i, j))  # 获取图片的pixel值??为什么是x,y,z??
            data.append([x / 256.0, y / 256.0, z / 256.0])  # ??为什么/256.0
    f.close()  # 关闭文件流
    return np.mat(data), m, n  # 分别返回三个数据,data,图片大小的纵值m和横值n
    # np.mat()方法将数据变为矩阵


imgData, row, col = loadData('bull.jpg')  # 调用自定义方法加载图片,获得三个返回值
km = KMeans(n_clusters=4)
# 加载KMeans算法, n_clusters指定了聚类中心个数为4
label = km.fit_predict(imgData)  # 聚类获得每个像素所属的类别
# fit_predict方法作用????
label = label.reshape([row, col])  # reshape方法????
pic_new = image.new("L", (row, col))  # 创建一张新的灰度图保存聚类后的结果
for i in range(row):
    for j in range(col):
        pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))  # 根据所属类别向图片添加灰度值
pic_new.save("result-bull-4.jpg", "JPEG")  # 以JPEG格式保存图片

# 通过设置不同的 k值,能够得到不同的聚类结果
# 同时k值的不确定也是Kmeans算法的一个缺点
# 为了达到好实验结果,需要进行多次尝试才能够选取最优的 k值
# 而像层次聚类的算法,就无需指定k值,只要给定限制条件,就能自动地得到类别数
