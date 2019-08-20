import time
import random
import numpy as np
import multiprocessing
import os
import tensorflow as tf
import pandas as pd
import math
# start1=time.time()
# s1 = [0 for i in range(0,1000000)]
# for i in range(1,10000):
#     s1[i]=[random.randint(0,1000)]
# stop1=time.time()
# print("使用+耗用的时间：",stop1-start1)
# start2=time.time()

# start2 = time.time()
# # label = np.zeros((3,2),float)
# label = []
# for j in range(27000000):
#     label.append(1)
# stop2=time.time()
# print("耗时：",stop2-start2)
# print()
#
# list1 = [0,0,0]
# list2 = ['a','b','c']
# print(list1+list2)

# def fun(i):
#     return i,i*i
#
# items = [x for x in range(3506)]
# pool = multiprocessing.Pool(4)
# res = pool.map(fun, items)
# pool.close()
# pool.join()
# print(res)
# print(len(res))

#
# import tensorflow as tf
#
# 两个矩阵相乘
# x = tf.constant([[[1.0, 7.0, 3.0,4.0], [4.0, 4.0, 5.0,2.0], [3.0, 6.0, 9.0,12.0]],
#                  [[1.0, 8.0, 3.0,16.0], [10.0, 4.0, 6.0,8.0], [3.0, 5.0, 9.0,10.0]],
#                 ])
# y = tf.constant([[3, 4, 5.,6.],[1, 2, 3,4]])
# y = tf.expand_dims(y, 1)
# # 注意这里这里x,y要有相同的数据类型，不然就会因为数据类型不匹配而出错
# print(x.get_shape())
# print(y.get_shape())
# z = tf.multiply(x, y)
# print(z.get_shape())
#
# with tf.Session() as sess:
#     print(sess.run(z))

# 正则化测试
# batch_size = 128
# sequence_len = 30
# embedding_len = 64
# K = 40

#
# x_emb_0 = np.random.rand(batch_size,sequence_len,embedding_len)
# W_class_tran = np.random.rand(embedding_len,K)
# topic_distribution = np.random.rand(K)
#
#
# x_emb_norm = tf.nn.l2_normalize(x_emb_0, axis = 2)  # b * s * e
#
# W_class_tran = tf.multiply(W_class_tran, topic_distribution) # 元素相乘，加入gamma
# W_class_norm = tf.nn.l2_normalize(W_class_tran, axis = 0)   # e * c
# G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c   # Ghat：归一化矩阵
#
# print(G)

#     x_emb_0 = tf.squeeze(x_emb, )  # b * s * e   # tf.squeeze：将原始张量中所有维度为1的那些维都删掉的结果
#     x_emb_norm = tf.nn.l2_normalize(x_emb_0, axis = 2)  # b * s * e
#     W_class_tran = tf.multiply(W_class_tran, opt.topic_distribution) # 元素相乘，加入gamma
#     W_class_norm = tf.nn.l2_normalize(W_class_tran, axis = 0)   # e * c
#     G = tf.contrib.keras.backend.dot(x_emb_norm, W_class_norm)  # b * s * c   # Ghat：归一化矩阵

# 正则化测试
# import tensorflow as tf
# input_data = tf.constant([[1.0,2,3],[4.0,5,6],[7.0,8,9],[10.0,11,12]])
# distribution = tf.constant([1.0,2,3])
# # output = tf.nn.l2_normalize(input_data, dim = 0)
# output = tf.multiply(input_data, distribution) # 元素相乘，加入gamma
# # output1 = tf.nn.l2_normalize(input_data, dim = 0)
# with tf.Session() as sess:
#     # print(sess.run(input_data))
#     print(sess.run(output))
#     # print(sess.run(output1))

# Z = np.zeros(3506, dtype='int')
# Z[3500] = 15
# Z[3239] = 38
# print(Z)
# statistic = pd.value_counts(Z, normalize=False)
# topic_distribution = np.zeros(40, dtype='float32')
# for i in statistic.index:
#     topic_distribution[i]=statistic[i]
# print(topic_distribution)
# print(np.exp(topic_distribution))
# # topic_distribution = np.exp(topic_distribution) / sum(np.exp(topic_distribution))
# # topic_distribution[:] = [x / 40 for x in topic_distribution]
# print(topic_distribution)
#

# def GetTopicDistribution():
#     # statistic = pd.value_counts(self.Z, normalize=False)
#     topic_distribution = np.zeros(40, dtype='float32')
#     randomres = [73.,59.,71.,80.,68.,50. ,  52.   ,65.,   77.  , 63.,   61. , 212.,
#      89.  ,125.,   93.,  112. , 108. ,  60. , 128.,  111. ,  68.,  102. , 120.,   92.,
#      97. , 109.  , 88. , 100.,   76.  , 63. ,  43.  , 91.,  168.  , 59. ,  70. ,  87.,
#      63.   ,68.  ,127. ,58.]
#     for i in range(40):
#         topic_distribution[i] = randomres[i]
#     # topic_distribution[:] = [x / self.K for x in topic_distribution]
#     topic_distribution = topic_distribution - np.max(topic_distribution)
#     topic_distribution = np.exp(topic_distribution) / sum(np.exp(topic_distribution))
#     print(topic_distribution)
#     return topic_distribution
# GetTopicDistribution()
# K = 40
# p = [1. / K] * K
# print(np.random.multinomial(1, p))
# topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
# print(topic)
# x = np.array([9.9e-01, 9.9e-01, 1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09,
#        1.0e-09, 1.0e-09, 1.0e-09], dtype=np.float32)
# y = x / x.sum()
# # np.random.multinomial(1, y)
# print(np.random.choice(len(y),p=y))


import pandas as pd
import re
from nltk import *
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections
import gensim

import codecs
import re
import _pickle as cPickle

from gensim.models.word2vec import Word2Vec
import _pickle

def getVocab():
    cols = [0] * 301
    cols[0] = 'word'
    for i in range(300):
        cols[i + 1] = i

    model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    google_list = list(model.vocab.keys())
    # print(google_list)
    print("Google list finished")
    # print(google_list.index('http'))
    # print(google_list.index('android'))

    vec_stanford = pd.read_csv('../data/glove.42B.300d.txt',names=cols,sep=' ')
    stanford_list = vec_stanford['word'].tolist()
    print("stanford list finished")
    # print(stanford_list.index('http'))
    # print(stanford_list.index('android'))
    return google_list,stanford_list

def preprocess_corpus(google_list,stanford_list):
    full_corpus = pd.read_csv('../data/full-corpus.csv')
    del full_corpus['Sentiment']
    del full_corpus['TweetId']
    del full_corpus['TweetDate']

    emnlp_dict = pd.read_csv('../data/emnlp_dict.txt',sep='	',names=['origin','true'])
    wordlist_origin = []
    wordlist_true = []
    for index,row in emnlp_dict.iterrows():
        wordlist_origin.append(row['origin'])
        wordlist_true.append(row['true'])
    frequency = collections.Counter([])

    df = full_corpus
    str_filtered_stopwords = []
    for index, row in df.iterrows():
        str = row['TweetText']
        str = str.lower() # 小写
        str = str.replace("apple", "")
        str = str.replace("twitter", "")
        str = str.replace("microsoft", "")
        str = str.replace("google", "")
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(str) # 分词
        line = []
        for word in words:
            word = re.sub(r'[^a-zA-Z]', '', word)  # 只保留英文
            if(word==''):
                continue
            else:
                if word in wordlist_origin: # 单词标准化
                    index = wordlist_origin.index(word)
                    word = wordlist_true[index]
                if len(word)>3 and word not in stopWords and word in google_list:
                    line.append(word)
        if(len(line)!=0):
            frequency_line = collections.Counter(line)
            frequency = frequency + frequency_line
            str_filtered_stopwords.append(line)
    print(frequency)
    print(len(frequency))
    removedic = []
    for key in frequency:
        if(frequency[key]<3):
            removedic.append(key)

    for line in str_filtered_stopwords:
        for word in line:
            if word in removedic:
                line.remove(word)
        if(len(line)==0):
            str_filtered_stopwords.remove(line)

    return str_filtered_stopwords

def Save_list(list1, filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')  # 相当于Tab一下，换一个单元格
        file2.write('\n')  # 写完一行立马换行
    file2.close()

def precess(filename,encoding,outputfilename): # copy from TWE/preprocess_ch.py
	docno_list=[]
	doccontent_list=[]
	doctext_list=[]
	f=codecs.open(filename,'r')
	ix=0
	max_len=0
	for line in f:
		doctext_list.append(line)
		token_list=line.split()
		if max_len<len(token_list):
			max_len=len(token_list)
		if len(token_list)!=0:
			doccontent_list.append(token_list)
			docno_list.append(ix)
			ix+=1
	f.close()
	print('docnumber:',ix)
	print('max_len:',max_len)
	vocab={}
	for doc in doccontent_list:
		for w in doc:
			if w in vocab:
				vocab[w]+=1
			else:
				vocab[w]=1
	vocab1={}
	for w in vocab:
		if vocab[w]>20:
			vocab1[w]=vocab[w]
	print('len_vocab:',len(vocab1))

	wordtoix = {}
	ixtoword = {}
	wordtoix['UNK'] = 0
	ixtoword[0] = 'UNK'
	ix = 1
	for w in vocab1:
		wordtoix[w] = ix
		ixtoword[ix] = w
		ix += 1

	doccententindex = []
	for doc in doccontent_list:
		docix_list = []
		for w in doc:
			if w in wordtoix:
				docix_list.append(wordtoix[w])
			else:
				docix_list.append(wordtoix['UNK'])
		doccententindex.append(docix_list)
	cPickle.dump([doccententindex,docno_list,wordtoix, ixtoword,doctext_list], open(outputfilename, "wb"))


if __name__=='__main__':
    # google_list,stanford_list = getVocab()
    corpus_filtered = preprocess_corpus()
    Save_list(corpus_filtered, 'tweet_filtered')

    # filename = '../data/corpus_title.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/news_train_title.p'
    # precess(filename, encoding, outputfilename)