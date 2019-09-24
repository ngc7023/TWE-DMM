"""
 @Time    : 2019/8/29 9:30
 @Author  : Pangjy
 @File    : TWE4/Classify.py
 @Software: PyCharm
 Based on Evalaution/Classify.py
 用于tweet分类
"""
from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import _pickle as cPickle
import codecs
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def classifier(data):
    train_x=data[0][0]
    train_y=data[0][1]
    # print(train_y)
    clf = SVC(kernel='rbf', probability=True)
    # clf=KNeighborsClassifier()
    clf.fit(train_x, train_y)
    test_x=data[1][0]
    test_y=data[1][1]
    test_predicted=clf.predict(test_x)
    # print(test_y)
    # print(test_predicted)
    accuracy = np.mean(test_predicted == test_y)

    print("The accuracy for classification of non-label text is %s" % accuracy)
    print(metrics.classification_report(test_y, test_predicted, digits=3))

def getdata3(datapath, thetafile):
    x = cPickle.load(open(datapath, "rb"),encoding='iso-8859-1')
    # train, val, test = x[0], x[1], x[2]
    # wordtoix, ixtoword = x[6], x[7]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    train_lab += val_lab
    train_lab_index = []
    test_lab_index = []
    for i in range(len(train_lab)):
        label_index = int(np.where(train_lab[i]>0)[0])
        train_lab_index.append(label_index)
    for i in range(len(test_lab)):
        label_index = int(np.where(test_lab[i]>0)[0])
        test_lab_index.append(label_index)
    docfeature_list = []
    f=codecs.open(thetafile, 'r')
    for line in f:
        tokens=line.strip().split()
        docfeature_list.append([float(t) for t in tokens])
        # if 'NaN' in tokens:
        #     docfeature_list.append(docfeature_list[-1])
        # else:
        #     docfeature_list.append([float(t) for t in tokens])
    f.close()
    print(len(train_lab))
    print(len(test_lab))
    print(len(docfeature_list))
    print(len(train_lab_index))
    print(len(test_lab_index))
    data_train = [docfeature_list[:len(train_lab)], train_lab_index]
    data_test = [docfeature_list[len(train_lab):], test_lab_index]
    return [data_train, data_test]

if __name__=='__main__':
    # ldathetafileprefix = '../RE/LDA/k_'
    # ldathetafilesuffix = '/testLDA.theta'
    # for k in [10, 20, 30, 40]:
    # 	print('lda model classify result: k=', k)
    # 	ldathetafile = ldathetafileprefix + str(k) + ldathetafilesuffix
    # 	data = getdata(datafile, ldathetafile)
    # 	classifier(data)
    #
    # dmmthetafileprefix = '../RE/DMM/k_'
    # dmmthetafilesuffix = '/testDMM.theta'
    # for k in [10, 20, 30, 40]:
    # 	print('dmm model classify result:k=', k)
    # 	dmmthetafile = dmmthetafileprefix + str(k) + dmmthetafilesuffix
    # 	data = getdata(datafile, dmmthetafile)
    # 	classifier(data)
    #
    # ldathetafileprefix='../RE/LFLDA/k_'
    # ldathetafilesuffix='/testLFLDA.theta'
    # for k in [10,20,30,40]:
    # 	print('lflda model classify result: k=',k)
    # 	ldathetafile=ldathetafileprefix+str(k)+ldathetafilesuffix
    # 	data=getdata(datafile,ldathetafile)
    # 	classifier(data)
    #
    # datafile='../data/classifydata/classifydata_index.p'
    # dmmthetafileprefix = '../RE/LFDMM/k_'
    # dmmthetafilesuffix = '/testLFDMM.theta'
    # for k in [40]:
    #     print('lfdmm model classify result:k=',k)
    #     dmmthetafile = dmmthetafileprefix + str(k) + dmmthetafilesuffix
    #     data = getdata(datafile, dmmthetafile)
    #     classifier(data)
    #
    # twethetafileprefix = '../RE/TWE_DMM/k_'
    # twethetafilesuffix = '/thetafile.txt'
    # for k in [10, 20, 30, 40]:
    # 	print('twedmm model classify result: k=', k)
    # 	twethetafile = twethetafileprefix + str(k) + twethetafilesuffix
    # 	data = getdata(datafile, twethetafile)
    # 	classifier(data)
    #
    # llathetafileprefix = '../RE/LLA_topic/k_'
    # llathetafilesuffix = '/thetafile.txt'
    # for k in [10, 20, 30, 40]:
    # 	print('lla model classify result: k=', k)
    # 	llathetafile = llathetafileprefix + str(k) + llathetafilesuffix
    # 	data = getdata(datafile, llathetafile)
    # 	classifier(data)
    #

    # llathetafileprefix = '../RE/LLA_word/k_'
    # llathetafilesuffix = '/thetafile.txt'
    # for k in [10, 20, 30, 40]:
    # 	print('lla model classify result: k=', k)
    # 	llathetafile = llathetafileprefix + str(k) + llathetafilesuffix
    # 	data = getdata(datafile, llathetafile)
    # 	classifier(data)

    # Tweet-LDA
    # for k in [10]:
    #     for proportion in [0.7]:
    #         thetafile = '../TWE5/re_LDA/re_tweet/'+str(k)+'/'+str(proportion)+'/thetafile.txt'
    #         datapath = '../DatasetProcess/2_Partition_Dataset_and_Generate_Embedding/outputdata/tweet_filtered' + str(proportion) + '.p'
    #         data = getdata3(datapath, thetafile)
    #         classifier(data)
    #
    # # Tweet-MDKLDA
    # for k in [10]:
    #     for proportion in [0.7]:
    #         thetafile = '../TWE5/re_MDKLDA/re_tweet/'+str(k)+'/'+str(proportion)+'/thetafile.txt'
    #         datapath = '../DatasetProcess/2_Partition_Dataset_and_Generate_Embedding/outputdata/tweet_filtered' + str(proportion) + '.p'
    #         data = getdata3(datapath, thetafile)
    #         classifier(data)


    # Tweet-LDA
    # for k in [10]:
    #     for proportion in [0.3]:
    #         thetafile = '../TWE5/re_LDA/re_N20short/'+str(k)+'/'+str(proportion)+'/thetafile.txt'
    #         datapath = '../DatasetProcess/2_Partition_Dataset_and_Generate_Embedding/outputdata/N20short' + str(proportion) + '.p'
    #         data = getdata3(datapath, thetafile)
    #         classifier(data)

    # Tweet-MDKLDA
    for k in [10]:
        for proportion in [0.7]:
            thetafile = '../TWE5/re_MDKLDA/re_N20short/'+str(k)+'/'+str(proportion)+'/thetafile.txt'
            datapath = '../DatasetProcess/2_Partition_Dataset_and_Generate_Embedding/outputdata/N20short' + str(proportion) + '.p'
            data = getdata3(datapath, thetafile)
            classifier(data)

