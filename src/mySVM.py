from numpy import *
import matplotlib.pyplot as plt

fileName = 'iris.data'


# 加载数据集
def loadDataSet(fileName):
    trainDataMat = []
    trainLabelMat = []
    testDataMat = []
    testLabelMat = []

    firstAttribute = 0
    secondAttribute = 2

    fr = open(fileName)
    num = 0
    for line in fr.readlines():
        if line == '\n':
            continue
        lineArray = line.split(',')
        if num < 40:
            trainDataMat.append([float(lineArray[firstAttribute]), float(lineArray[secondAttribute])])
            num += 1
        else:
            testDataMat.append([float(lineArray[firstAttribute]), float(lineArray[secondAttribute])])
            num += 1
            if num == 50:
                num = 0

    for i in range(40):
        trainLabelMat.append(1)
    for i in range(40):
        trainLabelMat.append(-1)
    for i in range(10):
        testLabelMat.append(1)
    for i in range(10):
        testLabelMat.append(-1)

    return trainDataMat, trainLabelMat, testDataMat, testLabelMat


# 返回一个0-m之间非i的随机数
def selectJrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clipAlpha(aj, H, L):
    if aj > H: return H
    if aj < L: return L
    return aj


class optStruts:
    def __init__(self, dataMatIn, labelMatIn, C, toler):
        self.X = dataMatIn
        self.Y = labelMatIn
        self.C = C
        self.toler = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        # 误差缓存，第一列为是否有效（0无效，1有效），第二列为缓存的值
        self.eCache = mat(zeros((self.m, 2)))


# 求k列的误差
def calcEk(os, k):
    fXk = float(multiply(os.alphas, os.Y).T * (os.X * os.X[k, :].T)) + os.b
    # 误差
    ek = fXk - os.Y[k]
    return ek


# 用于选择第二个α，也就是αj
# 保证选择的αj可以优化最大的步长，参考前面的公式labelMat[j] * (Ei-Ej)/eta
def selectJ(i, os, Ei):
    maxK = -1
    maxDeleteE = 0
    Ej = 0
    # 1标识设置缓存为有效
    os.eCache[i] = [1, Ei]
    # 构造非零E值所对应的alpha值构造的列表
    validateEcacheList = nonzero(os.eCache[:, 0].A)[0]
    if len(validateEcacheList) > 1:
        for k in validateEcacheList:
            # 配对优化，不能优化自己
            if k == i: continue
            # 挨个算误差
            Ek = calcEk(os, k)
            deltaE = abs(Ei - Ek)
            if deltaE > maxDeleteE:
                maxK = k
                maxDeleteE = deltaE
                Ej = Ek
        # 返回具有最大误差的α坐标和误差Ek
        return maxK, Ej
    else:
        # 都为零，说明是起步阶段，随便选个非i的运算即可。
        j = selectJrand(i, os.m)
        Ej = calcEk(os, j)
    return j, Ej


# 算k的误差，并设置为有效
def updateEk(os, k):
    Ek = calcEk(os, k)
    os.eCache[k] = [1, Ek]


def innerL(i, os):
    # 获取i的误差
    Ei = calcEk(os, i)
    if ((Ei * os.Y[i] < -os.toler) and (os.alphas[i] < os.C)) or \
            ((Ei * os.Y[i] > os.toler) and (os.alphas[i] > 0)):
        j, Ej = selectJ(i, os, Ei)
        alphas_i_old = os.alphas[i].copy()
        alphas_j_old = os.alphas[j].copy()
        if os.Y[i] != os.Y[j]:
            L = max(0, os.alphas[j] - os.alphas[i])
            H = min(os.C, os.C + os.alphas[j] - os.alphas[i])
        else:
            L = max(0, os.alphas[j] + os.alphas[i] - os.C)
            H = min(os.C, os.alphas[j] + os.alphas[i])
        if H == L: return 0
        eta = 2.0 * os.X[i, :] * os.X[j, :].T - os.X[i, :] * os.X[i, :].T - os.X[j, :] * os.X[j, :].T
        if eta >= 0: return 0
        os.alphas[j] -= os.Y[j] * (Ei - Ej) / eta
        os.alphas[j] = clipAlpha(os.alphas[j], H, L)
        updateEk(os, j)
        if (abs(os.alphas[j] - alphas_j_old) < 0.0001): return 0
        os.alphas[i] += os.Y[i] * os.Y[j] * (alphas_j_old - os.alphas[j])
        updateEk(os, i)
        b1 = os.b - Ei - os.Y[i] * (os.alphas[i] - alphas_i_old) * os.X[i, :] * os.X[i, :].T \
             - os.Y[j] * (os.alphas[j] - alphas_j_old) * os.X[i, :] * os.X[j, :].T
        b2 = os.b - Ej - os.Y[i] * (os.alphas[i] - alphas_i_old) * os.X[i, :] * os.X[j, :].T \
             - os.Y[j] * (os.alphas[j] - alphas_j_old) * os.X[j, :] * os.X[j, :].T
        if (0 < os.alphas[i]) and (os.alphas[i] < os.C):
            os.b = b1
        elif (0 < os.alphas[j]) and (os.alphas[j] < os.C):
            os.b = b2
        else:
            os.b = (b1 + b1) / 2.0
        return 1
    else:
        return 0


def smop(dataMatIn, labelMatIn, C, toler, maxInter):
    # 构造os辅助对象
    os = optStruts(mat(dataMatIn), mat(labelMatIn).transpose(), C, toler)
    iter = 0
    entireSet = True
    alphaPairsChanged = 0
    while (iter < maxInter) and (entireSet or (alphaPairsChanged > 0)):
        alphaPairsChanged = 0
        if entireSet:
            for i in range(os.m):
                alphaPairsChanged += innerL(i, os)
            iter += 1
        else:
            nonBoundIds = nonzero((os.alphas.A > 0) * (os.alphas.A < C))[0]
            for i in nonBoundIds:
                alphaPairsChanged += innerL(i, os)
            iter += 1
        if entireSet:
            entireSet = False
        elif alphaPairsChanged == 0:
            entireSet = True
    return os.b, os.alphas


# 根据alpha计算出w
def calcWs(alpha, dataArr, labelArr):
    dataMat = mat(dataArr)
    labelMat = mat(labelArr).transpose()
    m, n = shape(dataMat)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alpha[i] * labelMat[i], dataMat[i, :].T)
    return w


def plotBestFit(dataMat, labelMat, weights, b):
    dataArr = array(dataMat)

    X1 = []
    Y1 = []
    X2 = []
    Y2 = []

    for i in range(len(dataArr)):
        if int(labelMat[i]) == 1:
            X1.append(dataArr[i, 0])
            Y1.append(dataArr[i, 1])
        else:
            X2.append(dataArr[i, 0])
            Y2.append(dataArr[i, 1])

    ax = plt.subplot(111)
    ax.scatter(X1, Y1, c='red', marker='o')
    ax.scatter(X2, Y2, c='green', marker='*')
    x = arange(4, 8.5, 0.1)
    y = (-b - weights[0][0] * x) / weights[1][0]
    x.shape = (len(x), 1)
    y.shape = (len(x), 1)
    plt.plot(x, y)
    plt.xlabel('sepal length')
    plt.ylabel('petal width')
    plt.show()


def getDataAccuracy(dataMat, labelMat, w1, b1, w2, b2, w3, b3):
    dataMat = array(dataMat)
    num = len(dataMat)
    predictMat = []
    rightNum = 0
    for i in range(num):
        class1 = 0
        class2 = 0
        class3 = 0
        y1 = dataMat[i, :] * mat(w1) + b1
        if y1 > 0:
            class1 += 1
        else:
            class2 += 1
        y2 = dataMat[i, :] * mat(w2) + b2
        if y2 > 0:
            class2 += 1
        else:
            class3 += 1
        y3 = dataMat[i, :] * mat(w3) + b3
        if y3 > 0:
            class1 += 1
        else:
            class3 += 1

        if (class1 > class2 and class1 > class3):
            predictMat.append('Iris-setosa')
        elif (class2 > class1 and class2 > class3):
            predictMat.append('Iris-versicolor')
        elif (class3 > class1 and class3 > class2):
            predictMat.append('Iris-virginica')
        else:
            predictMat.append('Iris-virginica')

        if i < num / 3:
            if predictMat[i] == 'Iris-setosa':
                rightNum += 1
        elif i < num * 2 / 3:
            if predictMat[i] == 'Iris-versicolor':
                rightNum += 1
        elif i < num:
            if predictMat[i] == 'Iris-virginica':
                rightNum += 1

    if num == 120:
        print("The accuracy score of trainDataSet is %f" % (rightNum / num))
    elif num == 30:
        print("The accuracy score of testDataSet is %f" % (rightNum / num))

    class1 = array([[x[0], x[1]] for i, x in enumerate(dataMat) if i < num / 3])
    class2 = array([[x[0], x[1]] for i, x in enumerate(dataMat) if (i >= num / 3 and i < num * 2 / 3)])
    class3 = array([[x[0], x[1]] for i, x in enumerate(dataMat) if (i >= num * 2 / 3 and i < num)])

    classA = array([[x[0], x[1]] for i, x in enumerate(dataMat) if predictMat[i] == 'Iris-setosa'])
    classB = array([[x[0], x[1]] for i, x in enumerate(dataMat) if predictMat[i] == 'Iris-versicolor'])
    classC = array([[x[0], x[1]] for i, x in enumerate(dataMat) if predictMat[i] == 'Iris-virginica'])

    title1 = ''
    title2 = ''
    if num == 120:
        title1 = 'right classification of trainSet'
        title2 = 'train classification of trainSet'
    elif num == 30:
        title1 = 'right classification of testSet'
        title2 = 'train classification of testSet'

    ax = plt.subplot(121)
    ax.scatter(class1[:, 0], class1[:, 1], c="red", marker='o', label="Iris-setos")
    ax.scatter(class2[:, 0], class2[:, 1], c="green", marker='*', label="Iris-versicolor")
    ax.scatter(class3[:, 0], class3[:, 1], c="blue", marker='+', label="Iris-virginica")
    plt.title(title1)
    plt.xlabel('sepal length')
    plt.ylabel('petal width')
    plt.legend(loc=0)

    ax = plt.subplot(122)
    ax.scatter(classA[:, 0], classA[:, 1], c="red", marker='o', label="Iris-setos")
    ax.scatter(classB[:, 0], classB[:, 1], c="green", marker='*', label="Iris-versicolor")
    ax.scatter(classC[:, 0], classC[:, 1], c="blue", marker='+', label="Iris-virginica")
    plt.title(title2)
    plt.xlabel('sepal length')
    plt.ylabel('petal width')
    plt.legend(loc=0)

    plt.show()


if __name__ == '__main__':
    trainDataMat, trainLabelMat, testDataMat, testLabelMat = loadDataSet(fileName)

    b1, alpha1 = smop(trainDataMat[0: 80], trainLabelMat, 1.5, 0.001, 40)
    w1 = calcWs(alpha1, trainDataMat[0: 80], trainLabelMat)
    plotBestFit(trainDataMat[0: 80], trainLabelMat, w1, b1)

    b2, alpha2 = smop(trainDataMat[40: 120], trainLabelMat, 1.5, 0.001, 40)
    w2 = calcWs(alpha2, trainDataMat[40: 120], trainLabelMat)
    plotBestFit(trainDataMat[40: 120], trainLabelMat, w2, b2)

    b3, alpha3 = smop(trainDataMat[0: 40] + trainDataMat[80: 120], trainLabelMat, 1.5, 0.001, 40)
    w3 = calcWs(alpha3, trainDataMat[0: 40] + trainDataMat[80: 120], trainLabelMat)
    plotBestFit(trainDataMat[0: 40] + trainDataMat[80: 120], trainLabelMat, w3, b3)

    getDataAccuracy(trainDataMat, trainLabelMat, w1, b1, w2, b2, w3, b3)
    getDataAccuracy(testDataMat, testLabelMat, w1, b1, w2, b2, w3, b3)
