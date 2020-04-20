import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
import matplotlib.pyplot as plt


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


##计算欧氏距离
def euclidDistance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


##生成相似矩阵
def genSimilarMatrix(X):
    M = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            M[i][j] = 1.0 * euclidDistance(X[i], X[j])
            M[j][i] = M[i][j]
    return M


##生成邻接矩阵，采用全连接法
def genAdjacentMatrix(X, sigma=1.0):
    M = np.exp(- X / 2 / sigma / sigma)
    return M


##生成归一化拉普拉斯矩阵
def genNormLaplacianMatrix(X):
    degree = np.sum(X, axis=1)  ##求每个顶点的度
    laplacianMatrix = np.diag(degree) - X  ##求拉普拉斯矩阵
    sqrtDegreeMatrix = np.diag(1.0 / (degree ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)


##生成规范化特征矩阵
def genNormCharacteristicMatrix(X, k):
    w, v = np.linalg.eig(X)  ##求特征值和特征向量
    v = np.transpose(v)
    w = zip(w, range(len(w)))
    w = sorted(w, key=lambda w: w[0])
    z = []
    for i in range(k):
        z.append(v[w[i][1]])
    z = np.transpose(z)
    for i in range((len(z))):  ##特征矩阵规范化
        sum = np.sum(z[i] ** 2)
        for j in range(len(z[i])):
            z[i][j] = z[i][j] / np.sqrt(sum)
    return z


def getRandIndex(label_pred):
    ##生成原始数据的标签
    label = []
    for i in range(3):
        for j in range(50):
            label.append(i)

    ##计算兰德指数
    true = 0
    false = 0
    num = len(label_pred)
    for i in range(num):
        for j in range(i + 1, num):
            if (label[i] == label[j] and label_pred[i] == label_pred[j]):
                true += 1
            elif (label[i] != label[j] and label_pred[i] != label_pred[j]):
                false += 1
    print("rand index: ", (true + false) / ((num * (num - 1) / 2)))


def Show(data, k, label_pred):
    ##解决画图是的中文乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    ##绘制k-means结果
    x0 = data[label_pred == 0]
    x1 = data[label_pred == 1]
    x2 = data[label_pred == 2]
    ax = plt.subplot(111)
    ax.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
    ax.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
    ax.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
    plt.title("聚类结果显示")
    plt.legend(loc=2)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()


if __name__ == "__main__":
    k = 3
    data = genDataset()  ##获取数据
    similarMatrix = genSimilarMatrix(data)  ##生成相似矩阵
    adjacentMatrix = genAdjacentMatrix(similarMatrix)  ##生成邻接矩阵
    normLaplacianMatrix = genNormLaplacianMatrix(adjacentMatrix)  ##生成归一化拉普拉斯矩阵
    normCharacteristicMatrix = genNormCharacteristicMatrix(normLaplacianMatrix, k)  ##生成规范化特征矩阵
    # estimator = KMeans(n_clusters=3).fit(normCharacteristicMatrix)  ##K-means聚类
    # label_pred = estimator.labels_
    label_pred = GaussianMixture(n_components=3, covariance_type='full').fit(normCharacteristicMatrix).predict(normCharacteristicMatrix)
    print(label_pred)
    getRandIndex(label_pred)
    Show(data, k, label_pred)
