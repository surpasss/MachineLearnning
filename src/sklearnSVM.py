import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import svm
import matplotlib.pyplot as plt

if __name__ == '__main__':

    features = []
    labels = []
    with open("iris.data", mode="r") as rf:
        for line in rf:
            if line == '\n':
                continue
            features.append(list(map(float, line.split(",")[: -1])))
            labels.append(line.split(",")[-1])

    train_class1_features, test_class1_features, train_class1_labels, test_class1_labels = train_test_split(features[0: 50], labels[0: 50], test_size=0.20,
                                                                                random_state=0)
    train_class2_features, test_class2_features, train_class2_labels, test_class2_labels = train_test_split(features[50: 100], labels[50: 100], test_size=0.20,
                                                                                                           random_state=0)
    train_class3_features, test_class3_features, train_class3_labels, test_class3_labels = train_test_split(features[100: 150], labels[100: 150], test_size=0.20,
                                                                                                           random_state=0)
    train_features = train_class1_features + train_class2_features + train_class3_features
    test_features = test_class1_features + test_class2_features + test_class3_features
    train_labels = train_class1_labels + train_class2_labels + train_class3_labels
    test_labels = test_class1_labels + test_class2_labels + test_class3_labels

    classifer = svm.SVC()  # svm class
    classifer.fit(train_features, train_labels)  # training the svc model

    train_predict = classifer.predict(train_features)
    train_accuracy_score = accuracy_score(train_labels, train_predict)
    print("The accruacy score of trainSet is %f" % train_accuracy_score)

    train_class1 = np.array([[x[0], x[1]] for i, x in enumerate(train_features) if train_labels[i] == "Iris-setosa\n"])
    train_class2 = np.array([[x[0], x[1]] for i, x in enumerate(train_features) if train_labels[i] == "Iris-versicolor\n"])
    train_class3 = np.array([[x[0], x[1]] for i, x in enumerate(train_features) if train_labels[i] == "Iris-virginica\n"])

    train_classA = np.array([[x[0], x[1]] for i, x in enumerate(train_features) if train_predict[i] == "Iris-setosa\n"])
    train_classB = np.array([[x[0], x[1]] for i, x in enumerate(train_features) if train_predict[i] == "Iris-versicolor\n"])
    train_classC = np.array([[x[0], x[1]] for i, x in enumerate(train_features) if train_predict[i] == "Iris-virginica\n"])


    ax = plt.subplot(121)
    ax.scatter(train_class1[:, 0], train_class1[:, 1], c="red", marker='o', label="Iris-setos")
    ax.scatter(train_class2[:, 0], train_class2[:, 1], c="green", marker='*', label="Iris-versicolor")
    ax.scatter(train_class3[:, 0], train_class3[:, 1], c="blue", marker='+', label="Iris-virginica")
    plt.title("right classification of trainSet")
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=0)

    ax = plt.subplot(122)
    ax.scatter(train_classA[:, 0], train_classA[:, 1], c="red", marker='o', label="Iris-setos")
    ax.scatter(train_classB[:, 0], train_classB[:, 1], c="green", marker='*', label="Iris-versicolor")
    ax.scatter(train_classC[:, 0], train_classC[:, 1], c="blue", marker='+', label="Iris-virginica")
    plt.title("train classification of trainSet")
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=0)

    plt.show()


    test_predict = classifer.predict(test_features)
    test_accuracy_score = accuracy_score(test_labels, test_predict)
    print("The accruacy score of testSet is %f" % test_accuracy_score)

    test_class1 = np.array([[x[0], x[1]] for i, x in enumerate(test_features) if test_labels[i] == "Iris-setosa\n"])
    test_class2 = np.array([[x[0], x[1]] for i, x in enumerate(test_features) if test_labels[i] == "Iris-versicolor\n"])
    test_class3 = np.array([[x[0], x[1]] for i, x in enumerate(test_features) if test_labels[i] == "Iris-virginica\n"])

    test_classA = np.array([[x[0], x[1]] for i, x in enumerate(test_features) if test_predict[i] == "Iris-setosa\n"])
    test_classB = np.array([[x[0], x[1]] for i, x in enumerate(test_features) if test_predict[i] == "Iris-versicolor\n"])
    test_classC = np.array([[x[0], x[1]] for i, x in enumerate(test_features) if test_predict[i] == "Iris-virginica\n"])

    ax = plt.subplot(121)
    ax.scatter(test_class1[:, 0], test_class1[:, 1], c="red", marker='o', label="Iris-setos")
    ax.scatter(test_class2[:, 0], test_class2[:, 1], c="green", marker='*', label="Iris-versicolor")
    ax.scatter(test_class3[:, 0], test_class3[:, 1], c="blue", marker='+', label="Iris-virginica")
    plt.title("right classification of testSet")
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=0)

    ax = plt.subplot(122)
    ax.scatter(test_classA[:, 0], test_classA[:, 1], c="red", marker='o', label="Iris-setos")
    ax.scatter(test_classB[:, 0], test_classB[:, 1], c="green", marker='*', label="Iris-versicolor")
    ax.scatter(test_classC[:, 0], test_classC[:, 1], c="blue", marker='+', label="Iris-virginica")
    plt.title("test classification of testSet")
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend(loc=0)

    plt.show()
