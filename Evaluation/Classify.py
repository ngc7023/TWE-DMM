from sklearn.svm import SVC
from sklearn import metrics
import numpy as np
import _pickle as cPickle
import codecs

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


def classifier(data):
    train_x=data[0][0]
    train_y=data[0][1]
    train_name=data[0][2]
    # clf = SVC(kernel='rbf', probability=True)
    clf=KNeighborsClassifier()
    clf.fit(train_x, train_y)
    test_x=data[1][0]
    test_y=data[1][1]
    test_name=data[1][2]
    test_predicted=clf.predict(test_x)
    accuracy = np.mean(test_predicted == test_y)
    # print accuracy
    print("The accuracy of twenty_test is %s" % accuracy)
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

if __name__=='__main__':
    datafile='../data/classifydata/classifydata_index.p'

    # ldathetafileprefix = '../RE/LDA/k_'
    # ldathetafilesuffix = '/testLDA.theta'
    # for k in [10, 20, 30, 40]:
    # 	print('da model classify result: k=', k)
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
    dmmthetafileprefix = '../RE/LFDMM/k_'
    dmmthetafilesuffix = '/testLFDMM.theta'
    for k in [10, 20, 30, 40]:
        print('lfdmm model classify result:k=',k)
        dmmthetafile = dmmthetafileprefix + str(k) + dmmthetafilesuffix
        data = getdata(datafile, dmmthetafile)
        classifier(data)
    #
    # twethetafileprefix = '../RE/TWE_DMM/k_'
    # twethetafilesuffix = '/thetafile.txt'
    # for k in [10, 20, 30, 40]:
    # 	print('twedmm model classify result: k=', k)
    # 	twethetafile = twethetafileprefix + str(k) + twethetafilesuffix
    # 	data = getdata(datafile, twethetafile)
    # 	classifier(data)

    # llathetafileprefix = '../RE/LLA_topic/k_'
    # llathetafilesuffix = '/thetafile.txt'
    # for k in [10, 20, 30, 40]:
    # 	print('lla model classify result: k=', k)
    # 	llathetafile = llathetafileprefix + str(k) + llathetafilesuffix
    # 	data = getdata(datafile, llathetafile)
    # 	classifier(data)




