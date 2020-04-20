import numpy as np
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt

##获取数据
data = []
with open("iris.data", mode="r") as rf:
    for line in rf:
        if line == '\n':
            continue
        tempLine = line.split(",")[: -1]
        data.append(list(map(float, tempLine)))

##KMEANS聚类，获取标签
X = np.array(data)
label_pred = KMeans(n_clusters=3).fit(X).labels_

##生成原始数据的标签
label = []
for i in range(3):
    for j in range(50):
        label.append(i)

##计算兰德指数
true = 0
false = 0
num = len(data)
for i in range(num):
    for j in range(i + 1, num):
        if (label[i] == label[j] and label_pred[i] == label_pred[j]):
            true += 1
        elif (label[i] != label[j] and label_pred[i] != label_pred[j]):
            false += 1
print("rand index: ", (true + false) / ((num * (num - 1) / 2)))

##解决画图是的中文乱码问题
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

##数据可视化
plt.scatter(X[:, 0], X[:, 1], c=label)
plt.title("Iris数据集显示")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()

##绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
ax = plt.subplot(111)
ax.scatter(x0[:, 0], x0[:, 1], c="red", marker='o', label='label0')
ax.scatter(x1[:, 0], x1[:, 1], c="green", marker='*', label='label1')
ax.scatter(x2[:, 0], x2[:, 1], c="blue", marker='+', label='label2')
plt.title("k-means聚类结果显示")
plt.legend(loc=2)
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.show()
