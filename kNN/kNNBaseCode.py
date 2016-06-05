# -*- coding:utf-8 -*-
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
# kNN算法伪代码
# 1.计算已知类别数据集中的点与当前点之间的距离
# 2.按照距离递增次序排列
# 3.选取与当前点距离最小的k个点
# 4.确定前k个点所在类别的出现频率
# 5.返回前k个点出现频率最高的类别作为当前点的预测分类


# 通用函数，数据导入
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]  # shape返回一个列表，即矩阵的行列数
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet  # tile函数的作用是重复某个数组
    sqDiffMat = diffMat ** 2  # 矩阵中的每个数求平方
    sqDistances = sqDiffMat.sum(axis=1)  # axis=1横向求和，axis=0纵向求和
    distances = sqDistances ** 0.5  # 开平方
    sortedDistIndicies = distances.argsort()  # 这里返回的是索引，直接就得到了前k的数
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# file2matrix函数，处理输入格式问题


def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()  # readlines将整个文件一次读取到一个列表中
    numberOfLines = len(arrayOfLines)  # 读取文件的行数
    returnMat = zeros((numberOfLines, 3))  # 创建并返回一个number行，3列的NumPy矩阵
    classLabelVector = []  # 分类结果向量，解析文本数据到列表
    index = 0
    for line in arrayOfLines:
        line = line.strip()  # 去除开头和结尾处的空格
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 数值归一化函数autoNorm,自动将较大的数值特征值转化到0-1之间


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    # 这里是取列的最小值,而不是取行的最小值,这里的0表示的是按列取，所以这里得到的是1*3的矩阵，三个特征值都取了
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  # 矩阵的行数
    normDataSet = dataSet - tile(minVals, (m, 1))  # tile函数将变量内容复制成输入矩阵同样大小的矩阵
    # 特征值相除,newValue = (oldValue-min)/(max-min)
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

# 定义一个分类器测试函数


def datingClassTest(K):
    hoRatio = 0.10  # 选择测试数据的比例
    datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classfierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                    datingLabels[numTestVecs:m], )
        print "the classifier came back with: %d, the real result is: %d"\
            % (classfierResult, datingLabels[i])
        if(classfierResult != datingLabels[i]):
            errorCount += 1.0
    print "the total error rate is: %f" % (errorCount / float(numTestVecs))

# 测试一下基本数据


def start_BaseDataSet(point, k):
    group, labels = createDataSet()
    classifiedResult = classify0(point, group, labels, k)
    print classifiedResult

# 使用Matplotlib显示datingTestSet2.txt数据集


def showScatterFigureOfDataSet():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datingDataMat[:, 0], datingDataMat[
               :, 1], 15.0 * array(datingLabels), 15.0 * array(datingLabels))
    # 第一个参数是所有点的X坐标，第二个参数是所有点的Y坐标，第三个点是颜色，第四个是大小
    plt.show()

# 得到归一化的矩阵


def getNormMat():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)


# 约会网站测试函数
def classifyPerson(K):
    resultList = ['not at all', 'in small doses', 'in large dose']
    percentTats = float(raw_input(
        "percentage of time spent playing video games?"))
    ffMiles = float(raw_input(
        "frequent flier miles earned per year?"))
    iceCream = float(raw_input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classfierResult = classify0(
        (inArr - minVals) / ranges, normMat, datingLabels, K)
    print "You will probably like this Person:", \
        resultList[classfierResult - 1]
