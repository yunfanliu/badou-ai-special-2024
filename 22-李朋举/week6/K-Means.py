# coding: utf-8



import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像灰度颜色
img = cv2.imread('D:\cv_workspace\picture\lenna.png', 0)
print(img.shape)

# 获取图像高度、宽度
rows, cols = img.shape[:]

# 图像二维像素转换为一维
data = img.reshape((rows * cols, 1))
# np.float32是numpy库中一个数据类型，表示单精度浮点数，占用 4 个字节, 了确保数据对象是单精度浮点数类型
data = np.float32(data)

# 停止条件 (type,max_iter,epsilon)
'''
使用了 OpenCV 库中的TermCriteria结构体来定义停止条件。这个结构体包含了三个参数：
type：表示停止条件的类型。在你的例子中，使用了cv2.TERM_CRITERIA_EPS，这意味着当误差（error）达到某个阈值（epsilon）时，迭代将停止。
max_iter：表示最大迭代次数。在你的例子中，设置为 10 次迭代。
epsilon：表示误差阈值。在你的例子中，设置为 1.0。
当迭代次数达到max_iter，或者误差小于等于epsilon时，迭代将停止。
'''
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

# 设置标签
flags = cv2.KMEANS_RANDOM_CENTERS
'''
KMEANS_RANDOM_CENTERS表示初始质心是随机选择的，而不是使用默认的方式（即第一个样本作为第一个质心，然后计算其他质心与第一个质心的距离，选择距离最远的样本作为下一个质心）。
随机选择初始质心可以提高算法的鲁棒性，尤其在数据集具有不同分布或存在噪声的情况下。
'''

# K-Means聚类 聚集成4类
compactness, labels, centers = cv2.kmeans(data, 4, None, criteria, 10, flags)
'''
在OpenCV中，Kmeans()函数原型如下所示：
retval, bestLabels, centers = kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
    data表示聚类数据，最好是np.flloat32类型的N维点集
    K表示聚类类簇数
    bestLabels表示输出的整数数组，用于存储每个样本的聚类标签索引
    criteria表示迭代停止的模式选择，这是一个含有三个元素的元组型数。格式为（type, max_iter, epsilon）
        其中，type有如下模式：
         —–cv2.TERM_CRITERIA_EPS :精确度（误差）满足epsilon停止。
         —-cv2.TERM_CRITERIA_MAX_ITER：迭代次数超过max_iter停止。
         —-cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER，两者合体，任意一个满足结束。
    attempts表示重复试验kmeans算法的次数，算法返回产生的最佳结果的标签
    flags表示初始中心的选择，两种方法是cv2.KMEANS_PP_CENTERS ;和cv2.KMEANS_RANDOM_CENTERS
    centers表示集群中心的输出矩阵，每个集群中心为一行数据
'''

# 生成最终图像
dst = labels.reshape((img.shape[0], img.shape[1]))

# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']  # 字体设置为SimHei，这是一种中文字体

# 显示图像
titles = [u'原始图像', u'聚类图像']
images = [img, dst]
for i in range(2):
    plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray'),
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
