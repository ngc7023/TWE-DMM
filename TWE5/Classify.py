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
    train_name=data[0][2]
    clf = SVC(kernel='rbf', probability=True)
    # clf=KNeighborsClassifier()
    clf.fit(train_x, train_y)
    test_x=data[1][0]
    test_y=data[1][1]
    test_name=data[1][2]
    test_predicted=clf.predict(test_x)
    accuracy = np.mean(test_predicted == test_y)
    # print accuracy
    print("The accuracy of 20 percent data for test is %s" % accuracy)
    print(metrics.classification_report(test_y, test_predicted,digits=3))

def getdata(datatfile,thetafile):
    label_dic={'体育':0,'女性':1,'文学出版':2,'校园':3}
    x=cPickle.load(open(datatfile,'rb'))
    docindex_list=x[0]
    label_list=x[1]
    wordtoix=x[2]
    ixtoword=x[3]
    del x
    label_index=[label_dic[n] for n in label_list]
    docfeature_list=[]
    f=codecs.open(thetafile,'r')
    for line in f:
        tokens=line.strip().split()
        # docfeature_list.append([float(t) for t in tokens])
        if 'NaN' in tokens:
            docfeature_list.append(docfeature_list[-1])
        else:
            docfeature_list.append([float(t) for t in tokens])
    f.close()
    # print(len(docfeature_list),len(label_index),len(label_list))
    data_train=[docfeature_list[:3306],label_index[:3306],label_list[:3306]]
    data_test=[docfeature_list[3306:],label_index[3306:],label_list[3306:]]
    return [data_train,data_test]

def getdata2(labelfile,thetafile):
    # label_dic={'alt.atheism':0,'comp.graphics':1,
    #            'comp.os.ms-windows.misc':2,'comp.sys.ibm.pc.hardware':3,
    #            'comp.sys.mac.hardware':4,'comp.windows.x':5,
    #            'misc.forsale':6,'rec.autos':7,
    #            'rec.motorcycles':8,'rec.sport.baseball':9,
    #            'rec.sport.hockey':10,'sci.crypt':11,
    #            'sci.electronics':12,'sci.med':13,
    #            'sci.space':14,'soc.religion.christian':15,
    #            'talk.politics.guns':16,'talk.politics.mideast':17,
    #            'talk.politics.misc':18}
    # label_dic={'体育':0,'女性':1,'文学出版':2,'校园':3}

    # 只能处理label是顺序的文件
    x = pd.read_csv(labelfile,names=['label'])
    label_list = x['label'].values.tolist()
    tmp_list = list(set(label_list))
    tmp_list.sort(key=label_list.index)
    label_dic = dict(zip(tmp_list,list(range(0,len(tmp_list)))))

    # print(label_list)
    # x=cPickle.load(open(datatfile,'rb'))
    # docindex_list=x[0]
    # label_list=x[1]
    # wordtoix=x[2]
    # ixtoword=x[3]
    # del x

    label_index=[label_dic[n] for n in label_list]
    docfeature_list=[]
    f=codecs.open(thetafile,'r')
    for line in f:
        tokens=line.strip().split()
        # docfeature_list.append([float(t) for t in tokens])
        if 'NaN' in tokens:
            docfeature_list.append(docfeature_list[-1])
        else:
            docfeature_list.append([float(t) for t in tokens])
    f.close()

    number_list = []
    for key in label_dic.keys():
        number_list.append(label_list.count(key))

    idx = 0
    train_docfeature_list = []
    test_docfeature_list = []
    train_label_index = []
    test_label_index = []
    train_label_list = []
    test_label_list = []
    for num in number_list:
        point = round(num*0.8)
        train_docfeature_list += docfeature_list[idx:point+idx]
        train_label_index += label_index[idx:point+idx]
        train_label_list += label_list[idx:point+idx]

        test_docfeature_list += docfeature_list[point+idx:num+idx]
        test_label_index += label_index[point+idx:num+idx]
        test_label_list += label_list[point+idx:num+idx]

        idx+=num
    # print(len(train_label_list))
    # print(len(test_label_list))

    data_train = [train_docfeature_list,train_label_index,train_label_list]
    data_test = [test_docfeature_list,test_label_index,test_label_list]

    # print(len(docfeature_list),len(label_index),len(label_list))
    # data_train=[docfeature_list[:3306],label_index[:3306],label_list[:3306]]
    # data_test=[docfeature_list[3306:],label_index[3306:],label_list[3306:]]
    return [data_train,data_test]

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

    datafile='../data/classifydata2/langdetect_tweet_label.txt'
    for k in [10,20,30,40]:
        lfdmmthetafileprefix = '../RE2/LFDMM/glove/k_'
        lfdmmthetafilesuffix = '/Tweet' + str(k) + '.theta'
        print('lfdmm model classify result: k =', k)
        lfdmmthetafile = lfdmmthetafileprefix + str(k) + lfdmmthetafilesuffix
        data = getdata2(datafile, lfdmmthetafile)
        classifier(data)

