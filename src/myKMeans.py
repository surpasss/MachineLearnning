import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


##获取数据
def genDataset():
    data = []
    with open("iris.data", mode="r") as rf:
        for line in rf:
            if line == '\n':
                continue
            tempLine = line.split(",")[: -1]
            data.append(list(map(float, tempLine)))
    return np.array(data)


##初始化各类中心点，一开始随机从样本中选择k个点当做各类的中心
def initCentroid(data, k):
    num, dim = data.shape
    centpoint = np.zeros((k, dim))
    l = [x for x in range(num)]
    np.random.shuffle(l)  ##随机排序
    for i in range(k):
        index = l[i]
        centpoint[i] = data[index]
    return centpoint


##计算欧氏距离
def euclidDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


##进行Kmeans分类
def KMeans(data, k):
    num = np.shape(data)[0]  ##样本个数
    ##记录各样本的类信息，0:属于哪个类，1:距离该类中心点的距离
    cluster = np.zeros((num, 2))
    cluster[:, 0] = -1
    change = True  ##记录是否有样本改变分类
    cp = initCentroid(data, k)  ##初始化各类中心点
    while change:
        change = False
        ##遍历每一个样本
        for i in range(num):
            minDist = 9999.9
            minIndex = -1
            ##计算该样本距离每一个类中心点的距离，找到距离最近的中心点
            for j in range(k):
                dis = euclidDistance(cp[j], data[i])
                if dis < minDist:
                    minDist = dis
                    minIndex = j
            ##如果找到的类中心点非当前类，则改变该样本的分类
            if cluster[i, 0] != minIndex:
                change = True
                cluster[i, :] = minIndex, minDist
        ##根据样本重新分类，计算新的类中心点
        for j in range(k):
            pointincluster = data[[x for x in range(num) if cluster[x, 0] == j]]
            cp[j] = np.mean(pointincluster, axis=0)
    return cp, cluster


##计算兰德指数
def getRandIndex(cluster):
    label = []
    for i in range(3):
        for j in range(50):
            label.append(i)
    true = 0
    false = 0
    num = cluster.shape[0]
    for i in range(num):
        for j in range(i + 1, num):
            if (label[i] == label[j] and cluster[i, 0] == cluster[j, 0]):
                true += 1
            elif (label[i] != label[j] and cluster[i, 0] != cluster[j, 0]):
                false += 1
    print("rand index: ", (true + false) / ((num * (num - 1) / 2)))


##展示结果，各类类使用不同的颜色，中心点使用X表示
def Show(data, k, cp, cluster):
    num, dim = data.shape
    color = ['r', 'g', 'b', 'c', 'y', 'm', 'k']
    ##二维图
    for i in range(num):
        mark = int(cluster[i, 0])
        plt.plot(data[i, 0], data[i, 1], color[mark] + 'o')
    for i in range(k):
        plt.plot(cp[i, 0], cp[i, 1], color[i] + 'x')
    plt.show()
    ##三维图
    ax = plt.subplot(111, projection='3d')
    for i in range(num):
        mark = int(cluster[i, 0])
        ax.scatter(data[i, 0], data[i, 1], data[i, 2], c=color[mark])
    for i in range(k):
        ax.scatter(cp[i, 0], cp[i, 1], cp[i, 2], c=color[i], marker='x')
    plt.show()


if __name__ == "__main__":
    k = 3  ##分类个数
    data = genDataset()
    cp, cluster = KMeans(data, k)
    getRandIndex(cluster)
    Show(data, k, cp, cluster)
