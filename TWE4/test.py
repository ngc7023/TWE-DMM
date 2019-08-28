import time
import random
import numpy as np
import multiprocessing
import os
import tensorflow as tf
import pandas as pd
import math
import emoji
import re

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

# from langdetect import detect,detect_langs
# str_line = "check	video	live	ipads	video	memory	ipad	brilliant"
# res = str(detect_langs(str_line)[0])[:2]
#
# for i in range(10):
#     print(i)
#     print(str(detect_langs(str_line)[0])[:2])
#
# print(detect_langs(str_line))
# if(res!="en"):
#     if(res!='en'):
#         print(str_line)
#         print(detect_langs(str_line))

# from gensim.corpora import Dictionary
# from gensim.models import CoherenceModel
#
# record = []
# L = 40
# topics = []
# for i in range(40):
#     topics.append(str(i))
# cm = CoherenceModel(topics=topics, texts=texts, dictionary=_dict,
#                         window_size=10, coherence='c_uci', topn=L, processes=4)
# print(cm.get_coherence())


# from gensim.corpora.dictionary import Dictionary
# def print_dict(dic):
#     for key in dic:
#         print(key,dic[key])
# a = [[u'巴西',u'巴西',u'英格兰'],[u'巴西',u'西班牙',u'法国']]
# b = [u'巴西',u'巴西',u'比利时',u'法国',u'法国']
# # a用来构造词典
# dic = Dictionary(a)
# print(type(dic))
#
# df = pd.read_csv('../data/TACL-datasets/N20short.txt')
# ldf = df.values.tolist()
# print(len(ldf))
# res = []
# for line in ldf:
#     res.append(line[0].split())
# print(res)

# from palmetto.palmettopy.palmetto import *
# palmetto = Palmetto()
# words = ["cake", "apple", "banana", "cherry", "chocolate"]
# print(palmetto.get_coherence(words,coherence_type="npmi"))

# df = pd.read_csv('../data/TACL-datasets/N20short.txt')

# import pickle as cPickle
# embpath = '../data/TACL-datasets/N20short_emb.p'
# # embpath = "../data/classifydata/classifydata_emb.p"
#
# # embpath = '../data/TACL-datasets/N20.p'
# tmp = cPickle.load(open(embpath, 'rb'))[0]
# print(tmp)
# # W_emb = np.array(cPickle.load(open(embpath, 'rb'))[0], dtype='float32')

print(random.randint(0, 1))