# # -*- coding: utf-8 -*-
# """
#  @Time    : 2019/8/12 9:37
#  @Author  : Pangjy
#  @File    : LFDMM/DMMModel.py
#  @Software: PyCharm
#  Restore dmm model from the result of LFDMM, and calculate the topic_coherence
# """
# import numpy as np
# import random
# import codecs
# from collections import OrderedDict
# import math
# import _pickle as cPickle
# import pandas as pd
# from gensim.models import CoherenceModel
#
#
# class Document0(object):
#     def __init__(self):
#         self.words = []
#         self.length = 0
#         self.occurenceToIndexCount = []
#
#
# class Document(object):
#     def __init__(self):
#         self.words = []
#         self.length = 0
#
#
# # 把整个文档及单词构成vocabulary（不允许重复）
# class DataPreprocessing(object):
#     def __init__(self):
#         self.docs_count = 0  # 语料库中总文档数
#         self.words_count = 0  # 词汇表中总词数
#         # 保存每个文档d的信息，单词序列，以及length
#         self.docs = []
#         # 建立vocabulary表
#         self.word2id = {}
#         self.id2word = {}
#
#
# class DMMmodel(object):
#     def __init__(self, loadpath):
#         x_data = cPickle.load(open(loadpath, "rb"))
#         train = x_data[0]
#         print(len(train))
#         train_lab = []
#         wordtoix, ixtoword = x_data[2], x_data[3]
#         del x_data
#
#         dpre = DataPreprocessing()
#         dpre.docs_count = len(train)
#         dpre.words_count = len(wordtoix)
#         for i in range(len(train)):
#             doc = Document()
#             doc.words = train[i]
#             doc.length = len(train[i])
#
#             wordcount = {}
#             wordcount_list = []
#             for w in train[i]:
#                 if w in wordcount:
#                     wordcount[w] += 1
#                     wordcount_list.append(wordcount[w])
#                 else:
#                     wordcount[w] = 1
#                     wordcount_list.append(wordcount[w])
#             doc.occurenceToIndexCount = wordcount_list
#             dpre.docs.append(doc)
#         dpre.word2id = wordtoix
#         print(dpre.docs_count)
#
#         self.dpre = dpre  # 获取预处理参数（文档预处理实例）
#         # 模型参数
#         self.K = 40  # 主题个数
#         self.beta = 0.01  # 每个主题下词的狄利克雷分布先验参数beta（超参数）
#         self.alpha = 0.1  # 每个文档下主题的狄利克雷分布先验参数alpha（超参数）
#         self.lam = 0.5  # 决定参数
#         self.iter_times = 2  # 最大迭代次数
#         self.top_words_num = 20  # 每个主题特征词个数？？？？
#
#         # self.prob=None # embedding预测单词的概率
#
#         self.phifile = './re/phifile1.txt'  # 词-主题分布文件phi
#         self.thetafile = './re/thetafile1.txt'
#         self.topNfile = './re/topNfile1.txt'  # 每个主题topN词文件
#         self.tassginfile = './re/tassginfile1.txt'  # 最后分派结果文件
#
#         self.p = np.zeros(self.K)  # 概率向量double类型，存储采样的临时变量
#         self.E = np.zeros(self.K, dtype='int')  # 每个主题的文档数
#         self.ndsum = np.zeros(self.dpre.docs_count, dtype='int')  # 每个doc中词的总数
#         self.F = np.zeros(self.K, dtype='int')  # 每个topic词的数量
#         self.nw = np.zeros((self.dpre.words_count, self.K), dtype='int')  # 词word在主题topic上的数量
#
#         self.Z = np.zeros(self.dpre.docs_count, dtype='int')
#
#         # 随机分配主题类型，为每个文档中的各个单词随机分配主题
#         for x in range(self.dpre.docs_count):
#             topic = random.randint(0, self.K - 1)  # 随机取一个主题
#             self.Z[x] = topic  # 第x篇文档的主题为topic
#             self.E[topic] += 1  # 每个主题的文档数
#             self.ndsum[x] = self.dpre.docs[x].length
#             for y in range(self.dpre.docs[x].length):
#                 self.nw[self.dpre.docs[x].words[y]][topic] += 1
#                 self.F[topic] += 1
#
#         self.theta = np.array([0.0 for y in range(self.K)])  # 全数据集主题分布
#         self.phi = np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])  # 主题-词分布
#
#
# class DMMmodel1(object):
#     def __init__(self, loadpath, K, opt):
#         x_data = cPickle.load(open(loadpath, "rb"))
#         train = x_data[0]
#         train_lab = []
#         wordtoix = x_data[2]
#         idxtoword = x_data[3]
#         del x_data
#
#         dpre = DataPreprocessing()
#         dpre.docs_count = len(train)
#         dpre.words_count = len(wordtoix)
#         for i in range(len(train)):  # train[i]是某篇文档
#             doc = Document()
#             doc.words = train[i]
#             doc.length = len(train[i])
#             dpre.docs.append(doc)
#         dpre.word2id = wordtoix
#         dpre.id2word = idxtoword
#         del wordtoix
#         self.dpre = dpre  # 获取预处理参数（文档预处理实例）
#         # 模型参数
#         self.K = K  # 主题个数
#         self.beta = 0.1  # 每个主题下词的狄利克雷分布先验参数beta（超参数）
#         self.alpha = 0.1  # 每个文档下主题的狄利克雷分布先验参数alpha（超参数）
#         self.lam = 0.5  # 决定参数
#         self.iter_times = 100  # 最大迭代次数
#         self.top_words_num = 20  # 每个主题特征词个数？？？？
#         self.sample_num = 0
#
#         # topic word embedding 预测每个词的概率
#         self.prob = []
#         self.probIdx = []
#
#         self.phifile = opt.phifile  # 词-主题分布文件phi
#         self.thetafile = opt.thetafile
#         self.topNfile = opt.topNfile  # 每个主题topN词文件
#         self.tagassignfile = opt.tagassignfile  # 最后分派结果文件
#
#         self.p = np.zeros(self.K)  # 概率向量double类型，存储采样的临时变量
#         self.E = np.zeros(self.K, dtype='int')  # 每个主题的文档数
#         self.ndsum = np.zeros(self.dpre.docs_count, dtype='int')  # 每个doc中词的总数
#         self.F = np.zeros(self.K, dtype='int')  # 每个topic词的数量
#         self.nw = np.zeros((self.dpre.words_count, self.K), dtype='int')  # 词word在主题topic上的数量
#
#         if opt.restore:
#             print("Restore Z")
#             self.Z = self.Restore_Z()
#         else:
#             print("No restore, Initialize Z")
#             self.Z = np.zeros(self.dpre.docs_count, dtype='int')
#         self.theta = np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])  # 全数据集主题分布
#         self.phi = np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])  # 主题-词分布
#
#
#     def init_Z(self, Z):
#         self.Z = Z
#         if np.sum(self.Z) != 0:
#             for x in range(len(self.Z)):
#                 topic = self.Z[x]
#                 self.E[topic] += 1  # 每个主题的文档数
#                 self.ndsum[x] = self.dpre.docs[x].length
#                 for y in range(self.dpre.docs[x].length):
#                     self.nw[self.dpre.docs[x].words[y]][topic] += 1
#                     self.F[topic] += 1
#         else:
#             # 随机分配主题类型，为每个文档中的各个单词随机分配主题
#             for x in range(len(self.Z)):
#                 topic = random.randint(0, self.K - 1)  # 随机取一个主题
#                 self.Z[x] = topic  # 第x篇文档的主题为topic
#                 self.E[topic] += 1  # 每个主题的文档数
#                 self.ndsum[x] = self.dpre.docs[x].length
#                 for y in range(self.dpre.docs[x].length):
#                     self.nw[self.dpre.docs[x].words[y]][topic] += 1
#                     self.F[topic] += 1
#
#     def sampling(self, i):
#         # 换主题
#         topic = self.Z[i]  # 第i个文档的主题
#         self.E[topic] -= 1  # topic的文档数
#         for j in range(self.dpre.docs[i].length):
#             word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
#             self.nw[word][topic] -= 1
#             self.F[topic] -= 1
#
#         Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 唯一单词总数
#
#         # gibbs sample
#         self.p = (self.E + self.alpha)
#         for j in range(self.dpre.docs[i].length):
#             word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
#             self.p = self.p * (self.nw[word] + self.beta) / (self.F + Vbeta)
#         p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
#         topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
#
#         self.E[topic] += 1
#         for j in range(self.dpre.docs[i].length):
#             word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
#             self.nw[word][topic] += 1
#             self.F[topic] += 1
#
#         return topic
#
#     # @profile
#     def sampling1(self, i):
#         # 换主题
#         # print("before sampling:",self.E)
#         topic = self.Z[i]  # 第i个文档的主题
#         self.E[topic] -= 1  # topic的文档数
#         doc = self.dpre.docs[i]
#         for word in doc.words:
#             self.nw[word][topic] -= 1
#         self.F[topic] -= doc.length
#
#         Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 唯一单词总数
#
#         # gibbs sample
#         self.p = (self.E + self.alpha)
#         for word in doc.words:
#             self.p = self.p * (self.nw[word] + self.beta) / (self.F + Vbeta)
#         if (np.sum(self.p) == 0):
#             p = [1. / self.K] * self.K
#             topic = np.argmax(np.random.multinomial(1, [1. / self.K] * self.K))  # 以平均概率随机选择主题
#         else:
#             p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
#             topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
#         self.E[topic] += 1
#         for word in doc.words:
#             self.nw[word][topic] += 1
#         self.F[topic] += doc.length
#         return topic, p
#
#     # @profile
#     def sampling2(self, i):
#         # 换主题，减数字
#         topic = self.Z[i]  # 第i个文档的主题
#         self.E[topic] -= 1  # topic的文档数
#
#         doc = self.dpre.docs[i]
#         for word in doc.words:
#             self.nw[word][topic] -= 1
#         self.F[topic] -= doc.length
#         Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 词汇表单词总数
#
#         # gibbs sample
#         probIdx = self.probIdx_start[i]
#         self.p = (self.E + self.alpha)  # len(E) = K # doc-topic # 每个主题的文档数量+文档下主题的狄利克雷分布alpha
#         for word in doc.words[1:]:  # 这篇文档的单词长度
#             self.p = self.p * ((1 - self.lam) * (self.nw[word] + self.beta) / (self.F + Vbeta) + self.lam * self.prob[
#                 probIdx])  # F: 每个topic词的数量; prob shape = (sample_number, word_number)某个样本下一个词是word的概率;
#             probIdx += 1
#         if (np.sum(self.p) == 0):
#             p = [1. / self.K] * self.K
#             topic = np.argmax(np.random.multinomial(1, [1. / self.K] * self.K))  # 以概率q的分布随机选择主题
#         else:
#             p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
#             topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
#         self.E[topic] += 1
#         for word in doc.words:
#             self.nw[word][topic] += 1
#         self.F[topic] += doc.length
#         return topic, p
#
#     # 训练LDA模型
#     def est(self):
#         for x in range(self.iter_times):
#             for i in range(self.dpre.docs_count):
#                 topic = self.sampling(i)
#                 self.Z[i] = topic
#         self.save()
#
#     # 训练DMM模型
#     def est1(self, opt):
#         self.probIdx = 0
#         for i in range(self.dpre.docs_count):  # 总文档数
#             topic, p = self.sampling1(i)
#             self.Z[i] = topic
#             opt.topic_distribution[i] = p
#
#     def est2(self, opt):
#         self.probIdx = 0
#         for i in range(self.dpre.docs_count):  # 总文档数
#             topic, p = self.sampling2(i)
#             self.Z[i] = topic
#             opt.topic_distribution[i] = p
#
#     def est_parallel0(self, i):
#         topic = self.sampling1(i)
#         return topic
#
#     def est_parallel(self, i):
#         topic = self.sampling2(i)
#         return topic
#
#     def _theta(self):
#         self.theta = self.E + self.alpha
#         self.theta = np.array(self.theta / np.sum(self.theta))
#
#     def _theta1(self):  # 文章-主题分布
#         self.probIdx = 0
#         for i in range(self.dpre.docs_count):
#             Vbeta = self.dpre.words_count * self.beta
#             theta = self.E + self.alpha
#             for j in range(1, self.dpre.docs[i].length):  # default_(0,length)
#                 word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
#                 theta = theta * (
#                 (1 - self.lam) * (self.nw[word] + self.beta) / (self.F + Vbeta) + self.lam * self.prob[self.probIdx])
#                 self.probIdx += 1
#             self.theta[i] = theta  # [[K],[K],....] doc_count
#
#     def _theta2(self):  # 计算Topic_Distribution初始值
#         self.probIdx = 0
#         for i in range(self.dpre.docs_count):
#             Vbeta = self.dpre.words_count * self.beta
#             theta = self.E + self.alpha
#             for j in range(1, self.dpre.docs[i].length):  # default_(0,length)
#                 word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
#                 theta = theta * (self.nw[word] + self.beta) / (self.F + Vbeta)
#                 self.probIdx += 1
#             self.theta[i] = theta  # [[K],[K],....] doc_count
#
#     def _phi(self):
#         for i in range(self.K):
#             self.phi[i] = (self.nw.T[i] + self.beta) / (self.F[i] + self.dpre.words_count * self.beta)
#             self.phi[i] = np.array(self.phi[i] / np.sum(self.phi[i]))
#
#     def save(self):
#         self._phi()
#         self._theta()
#         # 保存phi词-主题分布
#         with codecs.open(self.phifile, 'w') as f:
#             for x in range(self.K):
#                 for y in range(self.dpre.words_count):
#                     f.write(str(self.phi[x][y]) + '\t')
#                 f.write('\n')
#         # 保存theta主题分布
#         with codecs.open(self.thetafile, 'w') as f:
#             for x in range(self.K):
#                 f.write(str(self.theta[x]) + '\t')
#             f.write('\n')
#         # 保存每个主题topic的topN词
#         with codecs.open(self.topNfile, 'w', 'utf-8') as f:
#             self.top_words_num = min(self.top_words_num, self.dpre.words_count)
#             for x in range(self.K):
#                 f.write('Topic' + str(x) + ': ')
#                 twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
#                 twords.sort(key=lambda i: i[1], reverse=True)
#                 for y in range(self.top_words_num):
#                     word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
#                     f.write(word + '(' + str(twords[y][1]) + ') ')
#                 f.write('\n')
#                 # f.write('\t' * 2 + str(twords[y][0]) + '\t' + str(twords[y][1]) + '\n')
#         # 保存最后退出时，文档的主题
#         with codecs.open(self.tagassignfile, 'w') as f:
#             for x in range(self.dpre.docs_count):
#                 f.write(str(x) + ':' + str(self.Z[x]) + '\t')
#
#     def save1(self):
#         self._phi()
#         self._theta1()
#         # 保存phi 主题-词分布 # K * [word_number]
#         with codecs.open(self.phifile, 'w') as f:
#             for x in range(self.K):
#                 for y in range(self.dpre.words_count):
#                     f.write(str(self.phi[x][y]) + ' ')
#                 f.write('\n')
#         # 保存theta 文档-主题分布 # docs * [K]
#         with codecs.open(self.thetafile, 'w') as f:
#             for x in range(self.dpre.docs_count):
#                 for y in range(self.K):
#                     f.write(str(self.theta[x][y]) + ' ')
#                 f.write('\n')
#             # 保存每个主题topic的topN词
#             with codecs.open(self.topNfile, 'w', 'utf-8') as f:
#                 self.top_words_num = min(self.top_words_num, self.dpre.words_count)
#                 for x in range(self.K):
#                     f.write('Topic' + str(x) + ': ')
#                     twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
#                     twords.sort(key=lambda i: i[1], reverse=True)
#                     for y in range(self.top_words_num):
#                         word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
#                         f.write(str(twords[y][0]) + '(' + word + '_' + str(twords[y][1]) + ') ')
#                     f.write('\n')
#         # 保存最后退出时，文档的主题
#         with codecs.open(self.tagassignfile, 'w') as f:
#             for x in range(self.dpre.docs_count):
#                 f.write(str(self.Z[x]) + '\n ')
#
#     def save2(self):
#         self._phi()
#         self._theta2()
#         # 保存phi 主题-词分布 # K * [word_number]
#         with codecs.open(self.phifile, 'w') as f:
#             for x in range(self.K):
#                 for y in range(self.dpre.words_count):
#                     f.write(str(self.phi[x][y]) + ' ')
#                 f.write('\n')
#         # 保存theta 文档-主题分布 # docs * [K]
#         with codecs.open(self.thetafile, 'w') as f:
#             for x in range(self.dpre.docs_count):
#                 for y in range(self.K):
#                     f.write(str(self.theta[x][y]) + ' ')
#                 f.write('\n')
#             # 保存每个主题topic的topN词
#             with codecs.open(self.topNfile, 'w', 'utf-8') as f:
#                 self.top_words_num = min(self.top_words_num, self.dpre.words_count)
#                 for x in range(self.K):
#                     f.write('Topic' + str(x) + ': ')
#                     twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
#                     twords.sort(key=lambda i: i[1], reverse=True)
#                     for y in range(self.top_words_num):
#                         word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
#                         f.write(str(twords[y][0]) + '(' + word + '_' + str(twords[y][1]) + ') ')
#                     f.write('\n')
#         # 保存最后退出时，文档的主题
#         with codecs.open(self.tagassignfile, 'w') as f:
#             for x in range(self.dpre.docs_count):
#                 f.write(str(self.Z[x]) + '\n ')
#
#     def getTopicCoherence(self):
#         self.top_words_num = min(self.top_words_num, self.dpre.words_count)
#         coherence = 0
#         for x in range(self.K):
#             twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
#             twords.sort(key=lambda i: i[1], reverse=True)
#             twords = [w[0] for w in twords[:self.top_words_num]]
#             for wi in range(len(twords)):
#                 for wj in range(wi + 1, len(twords)):
#                     countij = 0
#                     counti = 0
#                     countj = 0
#                     for di in range(self.dpre.docs_count):
#                         if twords[wi] in self.dpre.docs[di].words:
#                             counti += 1
#                             if twords[wj] in self.dpre.docs[di].words:
#                                 countij += 1
#                                 countj += 1
#                         else:
#                             if twords[wj] in self.dpre.docs[di].words:
#                                 countj += 1
#                     coherence += math.log((countij + 1) / (countj * counti) * self.dpre.docs_count)
#         return coherence / self.K
#
#     def GetTopicDistribution(self):
#         statistic = pd.value_counts(self.Z, normalize=False)
#         topic_distribution = np.zeros(self.K, dtype='float32')
#         for i in statistic.index:
#             topic_distribution[i] = statistic[i]
#         # print(topic_distribution)
#         topic_distribution[:] = [x / self.K for x in topic_distribution]
#         # topic_distribution = topic_distribution - np.max(topic_distribution)
#         # topic_distribution = np.exp(topic_distribution)/sum(np.exp(topic_distribution))
#
#         print("topic_distribution:", topic_distribution)
#
#     def Restore_Z(self):
#         data = pd.read_csv(self.tagassignfile, sep='\t', names=['topic'])
#         Stored_Z = data['topic'].values.tolist()
#         return Stored_Z
#
#     def initTopicEmb(self, all_wordemb):
#         self._phi()  # 更新词-主题分布 K*word_count
#         topic_emb = np.zeros([self.K, len(all_wordemb[0])], dtype='float32')
#         for i in range(self.K):
#             word_dis = self.phi[i]
#             word_dis = word_dis[:, np.newaxis]
#             topic_emb[i] = np.sum(np.multiply(word_dis, all_wordemb), axis=0)
#         return topic_emb
#
#
# class DMMmodel1(object):
#     def __init__(self, loadpath, K, opt):
#         x_data = cPickle.load(open(loadpath, "rb"))
#         train = x_data[0]
#         train_lab = []
#         wordtoix = x_data[2]
#         # idxtoword = x_data[3]
#         del x_data
#
#         dpre = DataPreprocessing()
#         dpre.docs_count = len(train)
#         dpre.words_count = len(wordtoix)
#         for i in range(len(train)):  # train[i]是某篇文档
#             doc = Document()
#             doc.words = train[i]
#             doc.length = len(train[i])
#             dpre.docs.append(doc)
#         dpre.word2id = wordtoix
#         del wordtoix
#         self.dpre = dpre  # 获取预处理参数（文档预处理实例）
#         # 模型参数
#         self.K = K  # 主题个数
#         self.beta = 0.1  # 每个主题下词的狄利克雷分布先验参数beta（超参数）
#         self.alpha = 0.1  # 每个文档下主题的狄利克雷分布先验参数alpha（超参数）
#         self.lam = 0.5  # 决定参数
#         self.iter_times = 100  # 最大迭代次数
#         self.top_words_num = 20  # 每个主题特征词个数？？？？
#         self.sample_num = 0
#
#         # topic word embedding 预测每个词的概率
#         self.prob = []
#         self.probIdx = []
#
#         self.phifile = opt.phifile  # 词-主题分布文件phi
#         self.thetafile = opt.thetafile
#         self.topNfile = opt.topNfile  # 每个主题topN词文件
#         self.tagassignfile = opt.tagassignfile  # 最后分派结果文件
#
#         self.p = np.zeros(self.K)  # 概率向量double类型，存储采样的临时变量
#         self.E = np.zeros(self.K, dtype='int')  # 每个主题的文档数
#         self.ndsum = np.zeros(self.dpre.docs_count, dtype='int')  # 每个doc中词的总数
#         self.F = np.zeros(self.K, dtype='int')  # 每个topic词的数量
#         self.nw = np.zeros((self.dpre.words_count, self.K), dtype='int')  # 词word在主题topic上的数量
#
#         if opt.restore:
#             print("Restore Z")
#             self.Z = self.Restore_Z()
#         else:
#             print("No restore, Initialize Z")
#             self.Z = np.zeros(self.dpre.docs_count, dtype='int')
#         self.theta = np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])  # 全数据集主题分布
#         self.phi = np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])  # 主题-词分布
#
#     def init_Z(self, Z):
#         self.Z = Z
#         if np.sum(self.Z) != 0:
#             for x in range(len(self.Z)):
#                 topic = self.Z[x]
#                 self.E[topic] += 1  # 每个主题的文档数
#                 self.ndsum[x] = self.dpre.docs[x].length
#                 for y in range(self.dpre.docs[x].length):
#                     self.nw[self.dpre.docs[x].words[y]][topic] += 1
#                     self.F[topic] += 1
#         else:
#             # 随机分配主题类型，为每个文档中的各个单词随机分配主题
#             for x in range(len(self.Z)):
#                 topic = random.randint(0, self.K - 1)  # 随机取一个主题
#                 self.Z[x] = topic  # 第x篇文档的主题为topic
#                 self.E[topic] += 1  # 每个主题的文档数
#                 self.ndsum[x] = self.dpre.docs[x].length
#                 for y in range(self.dpre.docs[x].length):
#                     self.nw[self.dpre.docs[x].words[y]][topic] += 1
#                     self.F[topic] += 1
#
#     def _theta(self):
#         self.theta = self.E + self.alpha
#         self.theta = np.array(self.theta / np.sum(self.theta))
#
#     def _theta2(self):  # 计算Topic_Distribution初始值
#         self.probIdx = 0
#         for i in range(self.dpre.docs_count):
#             Vbeta = self.dpre.words_count * self.beta
#             theta = self.E + self.alpha
#             for j in range(1, self.dpre.docs[i].length):  # default_(0,length)
#                 word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
#                 theta = theta * (self.nw[word] + self.beta) / (self.F + Vbeta)
#                 self.probIdx += 1
#             self.theta[i] = theta  # [[K],[K],....] doc_count
#
#     def _phi(self):
#         for i in range(self.K):
#             self.phi[i] = (self.nw.T[i] + self.beta) / (self.F[i] + self.dpre.words_count * self.beta)
#             self.phi[i] = np.array(self.phi[i] / np.sum(self.phi[i]))
#
#     def getTopicCoherence(self):
#         self.top_words_num = min(self.top_words_num, self.dpre.words_count)
#         coherence = 0
#         for x in range(self.K):
#             twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
#             twords.sort(key=lambda i: i[1], reverse=True)
#             twords = [w[0] for w in twords[:self.top_words_num]]
#             for wi in range(len(twords)):
#                 for wj in range(wi + 1, len(twords)):
#                     countij = 0
#                     counti = 0
#                     countj = 0
#                     for di in range(self.dpre.docs_count):
#                         if twords[wi] in self.dpre.docs[di].words:
#                             counti += 1
#                             if twords[wj] in self.dpre.docs[di].words:
#                                 countij += 1
#                                 countj += 1
#                         else:
#                             if twords[wj] in self.dpre.docs[di].words:
#                                 countj += 1
#                     coherence += math.log((countij + 1) / (countj * counti) * self.dpre.docs_count)
#         return coherence / self.K
#
#     def GetTopicDistribution(self):
#         statistic = pd.value_counts(self.Z, normalize=False)
#         topic_distribution = np.zeros(self.K, dtype='float32')
#         for i in statistic.index:
#             topic_distribution[i] = statistic[i]
#         topic_distribution[:] = [x / self.K for x in topic_distribution]
#         # topic_distribution = topic_distribution - np.max(topic_distribution)
#         # topic_distribution = np.exp(topic_distribution)/sum(np.exp(topic_distribution))
#
#         print("topic_distribution:", topic_distribution)
#
#     def Restore_Z(self):
#         data = pd.read_csv(self.tagassignfile, sep='\s+', names=['topic','1','2','3','4'])
#         Stored_Z = data['topic'].values.tolist()
#         return Stored_Z
#
#     def initTopicEmb(self, all_wordemb):
#         self._phi()  # 更新词-主题分布 K*word_count
#         topic_emb = np.zeros([self.K, len(all_wordemb[0])], dtype='float32')
#         for i in range(self.K):
#             word_dis = self.phi[i]
#             word_dis = word_dis[:, np.newaxis]
#             topic_emb[i] = np.sum(np.multiply(word_dis, all_wordemb), axis=0)
#         return topic_emb
#
#     def Gensim_getTopicCoherence(self):
#         topnwords = []
#         # get topn words
#         self._phi()
#         self.top_words_num = min(self.top_words_num, self.dpre.words_count)
#         topic_rec = []
#         for x in range(self.K):
#             twords = [(n, self.phi[x][n]) for n in range(1, self.dpre.words_count)]  # todo:topicN不够的时候怎么办
#             twords.sort(key=lambda i: i[1], reverse=True)  # 根据phi[x][n]排序
#             list = [self.dpre.id2word[wordid] for (wordid, num) in twords[0:self.top_words_num]]
#             for word in list:
#                 if (word == 'UNK'):
#                     topic_rec.append(x)
#             topnwords.append(list)
#         print(topnwords)
#         print(topic_rec)
#         try:
#             cm = CoherenceModel(topics=topnwords, texts=self.text, dictionary=self._dict,
#                                 window_size=10, coherence='c_uci', topn=self.top_words_num, processes=4)
#             cm2 = CoherenceModel(topics=topnwords, texts=self.text, dictionary=self._dict,
#                                  window_size=10, coherence='c_npmi', topn=self.top_words_num, processes=4)
#             print(cm.get_coherence(), cm2.get_coherence())
#         except:
#             print(topic_rec)
#             for x in topic_rec:
#                 print(topnwords[x])
#                 print("word sum:", self.F[x])
#                 print(self.nw.T[x])
#         return cm.get_coherence(), cm2.get_coherence()
#
# if __name__ == '__main__':
#     loadpath = '../data/news_train_title.p'
#     dmm = DMMmodel1(loadpath)
#     dmm.est1()
#     print(dmm.getTopicCoherence())

# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/13 14:30
 @Author  : Pangjy
 @File    : DMM/model2.py
 @Software: PyCharm
 Create this file based on LFDMM
"""
import numpy as np
import random
import codecs
from collections import OrderedDict
import math
import _pickle as cPickle
import pandas as pd
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary


class Document(object):
    def __init__(self):
        self.words = []
        self.length = 0


# 把整个文档及单词构成vocabulary（不允许重复）
class DataPreprocessing(object):
    def __init__(self):
        self.docs_count = 0  # 语料库中总文档数
        self.words_count = 0  # 词汇表中总词数
        # 保存每个文档d的信息，单词序列，以及length
        self.docs = []
        # 建立vocabulary表
        self.word2id = {}
        self.WordtopicAssignments = []


class DMMmodel(object):  # modify by Pangjy 08-13
    def __init__(self, loadpath, K, opt):
        self.corpus_path = opt.corpus_path
        self.phifile = opt.phifile  # 词-主题分布文件phi
        self.thetafile = opt.thetafile
        self.topNfile = opt.topNfile  # 每个主题topN词文件
        self.tagassignfile = opt.tagassignfile  # 最后分派结果文件

        x_data = cPickle.load(open(loadpath, "rb"))
        train = x_data[0]
        wordtoix, ixtoword = x_data[2], x_data[3]
        del x_data

        dpre = DataPreprocessing()
        dpre.docs_count = len(train)
        dpre.words_count = len(wordtoix)

        dpre.word2id = wordtoix
        dpre.id2word = ixtoword
        del wordtoix, ixtoword
        df = pd.read_csv(self.corpus_path, names=[''])
        list_df = df.values.tolist()
        # # print(dpre.docs_count)
        # # print(len(list_df))
        self.text = [0] * dpre.docs_count
        count_text = 0
        for line in list_df:
            self.text[count_text] = line[0].split()
            count_text += 1

        for i in range(len(train)):  # train[i]是某篇文档
            doc = Document()
            word_length = len(train[i])
            doc.words = train[i]
            # self.text[i] = list(map(lambda x: dpre.id2word[x], train[i]))
            # self.text[i] = train[i].apply(lambda x : self.ixtoword[x])
            doc.length = word_length
            doc.WordtopicAssignments = np.zeros(word_length, dtype='int')
            dpre.docs.append(doc)
        # print(self.text)
        self._dict = Dictionary(self.text)

        self.dpre = dpre  # 获取预处理参数（文档预处理实例）
        # 模型参数
        self.K = K  # 主题个数
        self.beta = 0.1  # 每个主题下词的狄利克雷分布先验参数beta（超参数）
        self.alpha = 0.1  # 每个文档下主题的狄利克雷分布先验参数alpha（超参数）
        self.lam = 0.5  # 决定参数
        self.iter_times = 100  # 最大迭代次数
        self.top_words_num = 20  # 每个主题特征词个数？？？？
        self.sample_num = 0

        # topic word embedding 预测每个词的概率
        self.prob = []
        self.probIdx = []

        self.p = np.zeros(self.K, dtype='float64')  # 概率向量double类型，存储采样的临时变量
        self.E = np.zeros(self.K, dtype='int')  # 每个主题的文档数
        self.ndsum = np.zeros(self.dpre.docs_count, dtype='int')  # 每个doc中词的总数
        self.F = np.zeros(self.K, dtype='int')  # 每个topic词的数量
        self.nw = np.zeros((self.dpre.words_count, self.K), dtype='int')  # 词word在主题topic上的数量

        self.topicWordCountDMM = np.zeros((self.dpre.words_count, self.K), dtype='int')
        # self.topicWordCountDMM = np.zeros((self.K,self.dpre.words_count),dtype='int')
        self.sumTopicWordCountDMM = np.zeros(self.K, dtype='int')
        # self.topicWordCountLF = np.zeros((self.K,self.dpre.words_count),dtype='int')
        self.topicWordCountLF = np.zeros((self.dpre.words_count, self.K), dtype='int')
        self.sumTopicWordCountLF = np.zeros(self.K, dtype='int')

        if opt.restore:
            print("Restore Z")
            self.Z = self.Restore_Z()
        else:
            print("No restore, Initialize Z")
            self.Z = np.zeros(self.dpre.docs_count, dtype='int')
        self.theta = np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])  # 全数据集主题分布
        self.phi = np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])  # 主题-词分布

    def init_Z(self, Z):
        self.Z = Z
        if np.sum(self.Z) != 0:  # 根据DMM的抽样结果恢复DMM模型
            print("Initilize Z with Assignment File")
            for x in range(len(self.Z)):
                topic = self.Z[x]
                self.E[topic] += 1  # 每个主题的文档数
                self.ndsum[x] = self.dpre.docs[x].length
                for y in range(self.dpre.docs[x].length):
                    self.nw[self.dpre.docs[x].words[y]][topic] += 1
                    self.F[topic] += 1
        else:
            # 随机分配主题类型，为每个文档中的各个单词随机分配主题
            print("Random Z")
            for x in range(len(self.Z)):
                topic = random.randint(0, self.K - 1)  # 随机取一个主题
                self.Z[x] = topic  # 第x篇文档的主题为topic
                self.E[topic] += 1  # 每个主题的文档数

                doc = self.dpre.docs[x]
                self.ndsum[x] = doc.length
                self.F[topic] += doc.length
                for y in range(self.dpre.docs[x].length):
                    self.nw[self.dpre.docs[x].words[y]][topic] += 1

                for i in range(doc.length):  # 根据LFDMM修改初始化方法
                    word = doc.words[i]
                    subtopic = random.randint(0, 1)
                    if (subtopic == 0):
                        self.sumTopicWordCountDMM[topic] += 1
                        self.topicWordCountDMM[word][topic] += 1
                        doc.WordtopicAssignments[i] = topic + self.K
                    else:
                        self.sumTopicWordCountLF[topic] += 1
                        self.topicWordCountLF[word][topic] += 1
                        doc.WordtopicAssignments[i] = topic

    def sampling(self, i):
        # 换主题
        topic = self.Z[i]  # 第i个文档的主题
        self.E[topic] -= 1  # topic的文档数
        for j in range(self.dpre.docs[i].length):
            word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
            self.nw[word][topic] -= 1
            self.F[topic] -= 1

        Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 唯一单词总数

        # gibbs sample
        self.p = (self.E + self.alpha)
        for j in range(self.dpre.docs[i].length):
            word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
            self.p = self.p * (self.nw[word] + self.beta) / (self.F + Vbeta)
        p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
        topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题

        self.E[topic] += 1
        for j in range(self.dpre.docs[i].length):
            word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
            self.nw[word][topic] += 1
            self.F[topic] += 1

        return topic

    # @profile
    def sampling1(self, i):
        # 换主题
        # print("before sampling:",self.E)
        topic = self.Z[i]  # 第i个文档的主题
        self.E[topic] -= 1  # topic的文档数
        doc = self.dpre.docs[i]
        for word in doc.words:
            self.nw[word][topic] -= 1
        self.F[topic] -= doc.length

        Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 唯一单词总数

        # gibbs sample
        self.p = (self.E + self.alpha)
        for word in doc.words:
            self.p = self.p * (self.nw[word] + self.beta) / (self.F + Vbeta)
        if (np.sum(self.p) == 0):
            p = [1. / self.K] * self.K
            topic = np.argmax(np.random.multinomial(1, [1. / self.K] * self.K))  # 以平均概率随机选择主题
        else:
            p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
            topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
        self.E[topic] += 1
        for word in doc.words:
            self.nw[word][topic] += 1
        self.F[topic] += doc.length
        return topic, p

    # @profile
    def sampling2(self, i):
        # 换主题，减数字
        topic = self.Z[i]  # 第i个文档的主题
        self.E[topic] -= 1  # topic的文档数

        doc = self.dpre.docs[i]
        for word in doc.words:
            self.nw[word][topic] -= 1
        self.F[topic] -= doc.length
        Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 词汇表单词总数

        # gibbs sample
        probIdx = self.probIdx_start[i]
        self.p = (self.E + self.alpha)  # len(E) = K # doc-topic # 每个主题的文档数量+文档下主题的狄利克雷分布alpha
        for word in doc.words[1:]:  # 这篇文档的单词长度
            self.p = self.p * ((1 - self.lam) * (self.nw[word] + self.beta) / (self.F + Vbeta) + self.lam * self.prob[
                probIdx])  # F: 每个topic词的数量; prob shape = (sample_number, word_number)某个样本下一个词是word的概率;
            # nw[word]要换成
            probIdx += 1
        if (np.sum(self.p) == 0):
            p = [1. / self.K] * self.K
            topic = np.argmax(np.random.multinomial(1, [1. / self.K] * self.K))  # 以概率p的分布随机选择主题
        else:
            p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
            topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
        self.E[topic] += 1
        for word in doc.words:
            self.nw[word][topic] += 1
        self.F[topic] += doc.length
        return topic, p

    # 训练LDA模型
    def est(self):
        for x in range(self.iter_times):
            for i in range(self.dpre.docs_count):
                topic = self.sampling(i)
                self.Z[i] = topic
        self.save()

    # 训练DMM模型
    def est1(self, opt):
        self.probIdx = 0
        for i in range(self.dpre.docs_count):  # 总文档数
            topic, p = self.sampling1(i)
            self.Z[i] = topic
            opt.topic_distribution[i] = p

    def est2(self, opt):
        self.probIdx = 0
        for i in range(self.dpre.docs_count):  # 总文档数
            topic, p = self.sampling2(i)
            self.Z[i] = topic
            opt.topic_distribution[i] = p

    def _theta(self):
        self.theta = self.E + self.alpha
        self.theta = np.array(self.theta / np.sum(self.theta))

    def _theta1(self):  # 文章-主题分布
        self.probIdx = 0
        for i in range(self.dpre.docs_count):
            Vbeta = self.dpre.words_count * self.beta
            theta = self.E + self.alpha
            for j in range(1, self.dpre.docs[i].length):  # default_(0,length)
                word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
                theta = theta * (
                (1 - self.lam) * (self.nw[word] + self.beta) / (self.F + Vbeta) + self.lam * self.prob[self.probIdx])
                self.probIdx += 1
            self.theta[i] = theta  # [[K],[K],....] doc_count

    def _theta2(self):  # 计算Topic_Distribution初始值
        self.probIdx = 0
        for i in range(self.dpre.docs_count):
            Vbeta = self.dpre.words_count * self.beta
            theta = self.E + self.alpha
            for j in range(1, self.dpre.docs[i].length):  # default_(0,length)
                word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
                theta = theta * (self.nw[word] + self.beta) / (self.F + Vbeta)
                self.probIdx += 1
            self.theta[i] = theta  # [[K],[K],....] doc_count

    def _phi(self):
        for i in range(self.K):
            self.phi[i] = (self.nw.T[i] + self.beta) / (self.F[i] + self.dpre.words_count * self.beta)
            self.phi[i] = np.array(self.phi[i] / np.sum(self.phi[i]))

    def save(self):
        self._phi()
        self._theta()
        # 保存phi词-主题分布
        with codecs.open(self.phifile, 'w') as f:
            for x in range(self.K):
                for y in range(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + '\t')
                f.write('\n')
        # 保存theta主题分布
        with codecs.open(self.thetafile, 'w') as f:
            for x in range(self.K):
                f.write(str(self.theta[x]) + '\t')
            f.write('\n')
        # 保存每个主题topic的topN词
        with codecs.open(self.topNfile, 'w', 'utf-8') as f:
            self.top_words_num = min(self.top_words_num, self.dpre.words_count)
            for x in range(self.K):
                f.write('Topic' + str(x) + ': ')
                twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
                twords.sort(key=lambda i: i[1], reverse=True)
                for y in range(self.top_words_num):
                    word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                    f.write(word + '(' + str(twords[y][1]) + ') ')
                f.write('\n')
                # f.write('\t' * 2 + str(twords[y][0]) + '\t' + str(twords[y][1]) + '\n')
        # 保存最后退出时，文档的主题
        with codecs.open(self.tagassignfile, 'w') as f:
            for x in range(self.dpre.docs_count):
                f.write(str(x) + ':' + str(self.Z[x]) + '\t')

    def save1(self):
        self._phi()
        self._theta1()
        # 保存phi 主题-词分布 # K * [word_number]
        with codecs.open(self.phifile, 'w') as f:
            for x in range(self.K):
                for y in range(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + ' ')
                f.write('\n')
        # 保存theta 文档-主题分布 # docs * [K]
        with codecs.open(self.thetafile, 'w') as f:
            for x in range(self.dpre.docs_count):
                for y in range(self.K):
                    f.write(str(self.theta[x][y]) + ' ')
                f.write('\n')
            # 保存每个主题topic的topN词
            with codecs.open(self.topNfile, 'w', 'utf-8') as f:
                self.top_words_num = min(self.top_words_num, self.dpre.words_count)
                for x in range(self.K):
                    f.write('Topic' + str(x) + ': ')
                    twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
                    twords.sort(key=lambda i: i[1], reverse=True)
                    for y in range(self.top_words_num):
                        word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                        f.write(str(twords[y][0]) + '(' + word + '_' + str(twords[y][1]) + ') ')
                    f.write('\n')
        # 保存最后退出时，文档的主题
        with codecs.open(self.tagassignfile, 'w') as f:
            for x in range(self.dpre.docs_count):
                f.write(str(self.Z[x]) + '\n ')

    def save2(self):
        self._phi()
        self._theta2()
        # 保存phi 主题-词分布 # K * [word_number]
        with codecs.open(self.phifile, 'w') as f:
            for x in range(self.K):
                for y in range(self.dpre.words_count):
                    f.write(str(self.phi[x][y]) + ' ')
                f.write('\n')
        # 保存theta 文档-主题分布 # docs * [K]
        with codecs.open(self.thetafile, 'w') as f:
            for x in range(self.dpre.docs_count):
                for y in range(self.K):
                    f.write(str(self.theta[x][y]) + ' ')
                f.write('\n')
            # 保存每个主题topic的topN词
            with codecs.open(self.topNfile, 'w', 'utf-8') as f:
                self.top_words_num = min(self.top_words_num, self.dpre.words_count)
                for x in range(self.K):
                    f.write('Topic' + str(x) + ': ')
                    twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
                    twords.sort(key=lambda i: i[1], reverse=True)
                    for y in range(self.top_words_num):
                        word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
                        f.write(str(twords[y][0]) + '(' + word + '_' + str(twords[y][1]) + ') ')
                    f.write('\n')
        # 保存最后退出时，文档的主题
        with codecs.open(self.tagassignfile, 'w') as f:
            for x in range(self.dpre.docs_count):
                f.write(str(self.Z[x]) + '\n ')

    def Gensim_getTopicCoherence(self):
        topnwords = []
        # get topn words
        self._phi()
        self.top_words_num = min(self.top_words_num, self.dpre.words_count)
        for x in range(self.K):
            twords = [(n, self.phi[x][n]) for n in range(1, self.dpre.words_count)]  # todo:topicN不够的时候怎么办
            twords.sort(key=lambda i: i[1], reverse=True)  # 根据phi[x][n]排序
            twords = twords[0:self.top_words_num]
            num = 0.1 / (self.F[x] + self.dpre.words_count * 0.1)
            num = float('%.19f' % num)
            # (1, 0.0001225790634959549) 表示N20small此处开始无有效的topNword
            if((1, num) in twords):
                # print("!")
                twords = twords[:twords.index((1, num))]
            if(len(twords)>1):
                list = [self.dpre.id2word[wordid] for (wordid, num) in twords[0:self.top_words_num]]
                topnwords.append(list)
        print("valid topic number:",len(topnwords))
        cm = CoherenceModel(topics=topnwords, texts=self.text, dictionary=self._dict,
                                window_size=10, coherence='c_uci', topn=self.top_words_num, processes=4)
        cm2 = CoherenceModel(topics=topnwords, texts=self.text, dictionary=self._dict,
                                 window_size=10, coherence='c_npmi', topn=self.top_words_num, processes=4)
        return cm.get_coherence(), cm2.get_coherence()

    def getTopicCoherence(self):
        empty_topic = 0
        self.top_words_num = min(self.top_words_num, self.dpre.words_count)
        coherence = 0
        npmi_coherence = 0
        for x in range(self.K):
            if (self.F[x] == 0):
                empty_topic += 1
                continue
            twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
            twords.sort(key=lambda i: i[1], reverse=True)
            twords_origin = twords[:self.top_words_num]
            twords = [w[0] for w in twords[:self.top_words_num]]
            for wi in range(len(twords)):
                for wj in range(wi + 1, len(twords)):
                    countij = 0
                    counti = 0
                    countj = 0
                    for di in range(self.dpre.docs_count):
                        if twords[wi] in self.dpre.docs[di].words:
                            counti += 1
                            if twords[wj] in self.dpre.docs[di].words:
                                countij += 1
                                countj += 1
                        else:
                            if twords[wj] in self.dpre.docs[di].words:
                                countj += 1
                    try:
                        # todo: modify PMI and NPMI ing
                        # PMI: m_lr(S_i) = log[(P(W', W*) + e) / (P(W') * P(W*))]
                        # NPMI: m_nlr(S_i) = m_lr(S_i) / -log[P(W', W*) + e]
                        if (counti == 0 or countj == 0):
                            continue
                        else:
                            tmp = math.log(
                                self.dpre.docs_count * (countij + 1.0e-12 * self.dpre.docs_count) / (counti * countj))
                            coherence += tmp
                            npmi_coherence += tmp / (-math.log(countij / self.dpre.docs_count + 1.0e-12))
                            # todo: counti == 0 空topic 或 topic的词数<topn
                    except:
                        print("exception")
                        print(wi, wj, counti, countj)
                        print(twords_origin)
                        print(x)
                        print(self.nw.T[x])
                        print(twords_origin[wi])
                        print(twords_origin[wj])
                        continue
        print("empty topic:", empty_topic)
        return coherence / (self.K - empty_topic), npmi_coherence / (self.K - empty_topic)

    def Restore_Z(self):

        data = pd.read_csv(self.tagassignfile, sep='\t', names=['topic'])
        Stored_Z = data['topic'].values.tolist()
        return Stored_Z

        cols = [0] * 1000
        for i in range(1000):
            cols[i] = i

        data = pd.read_csv(self.tagassignfile, sep=' ',names=cols)
        Stored_Z = data[0].values.tolist()
        print(Stored_Z)
        return Stored_Z

    def initTopicEmb(self, all_wordemb):
        self._phi()  # 更新词-主题分布 K*word_count
        topic_emb = np.zeros([self.K, len(all_wordemb[0])], dtype='float32')
        for i in range(self.K):
            word_dis = self.phi[i]
            word_dis = word_dis[:, np.newaxis]
            topic_emb[i] = np.sum(np.multiply(word_dis, all_wordemb), axis=0)
        return topic_emb

    def sampleSingleInitialIteration(self, opt):
        for i in range(self.dpre.docs_count):
            doc = self.dpre.docs[i]
            topic = self.Z[i]
            Vbeta = self.dpre.words_count * self.beta  # beta * vocabularySize

            self.E[topic] -= 1
            self.F[topic] -= doc.length

            for j in range(doc.length):
                word = doc.words[j]
                self.nw[word][topic] -= 1

                subtopic = doc.WordtopicAssignments[j]
                if (topic == subtopic):
                    self.topicWordCountLF[word][topic] -= 1
                    self.sumTopicWordCountLF[topic] -= 1
                else:
                    self.topicWordCountDMM[word][topic] -= 1;
                    self.sumTopicWordCountDMM[topic] -= 1;

            self.p = self.E + self.alpha
            for word in doc.words:
                self.p = self.p * (
                self.lam * (self.topicWordCountLF[word] + self.beta) / (self.sumTopicWordCountLF + Vbeta)
                + (1 - self.lam) * (self.topicWordCountDMM[word] + self.beta) / (self.sumTopicWordCountDMM + Vbeta))

            if (np.sum(self.p) == 0):
                dist = [1. / self.K] * self.K
                topic = np.argmax(np.random.multinomial(1, dist))  # 以平均概率随机选择主题
            else:
                dist = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
                choices = range(len(dist))
                topic = np.random.choice(choices, p=dist)
            # topic = np.argmax(np.random.multinomial(1, p))

            self.E[topic] += 1
            self.F[topic] += doc.length

            for j in range(doc.length):
                word = doc.words[j]
                self.nw[word][topic] += 1

                subtopic = topic
                if ((self.lam * (self.topicWordCountLF[word][topic] + self.beta) / (
                    self.sumTopicWordCountLF[topic] + Vbeta))
                        > ((1 - self.lam) * (self.topicWordCountDMM[word][topic] + self.beta) / (
                        self.sumTopicWordCountDMM[topic] + Vbeta))):
                    self.topicWordCountLF[word][topic] += 1
                    self.sumTopicWordCountLF[topic] += 1
                else:
                    self.topicWordCountDMM[word][topic] += 1
                    self.sumTopicWordCountDMM[topic] += 1
                    subtopic += self.K
                doc.WordtopicAssignments[j] = subtopic

            self.Z[i] = topic
            opt.topic_distribution[i] = dist

    def sampleSingleIteration(self, opt):
        probIdx = 0
        for i in range(self.dpre.docs_count):
            doc = self.dpre.docs[i]
            topic = self.Z[i]
            Vbeta = self.dpre.words_count * self.beta  # beta * vocabularySize

            self.E[topic] -= 1
            self.F[topic] -= doc.length

            for j in range(doc.length):
                word = doc.words[j]
                self.nw[word][topic] -= 1

                subtopic = doc.WordtopicAssignments[j]
                if (topic == subtopic):
                    self.topicWordCountLF[word][topic] -= 1
                    self.sumTopicWordCountLF[topic] -= 1
                else:
                    self.topicWordCountDMM[word][topic] -= 1;
                    self.sumTopicWordCountDMM[topic] -= 1;

            prob_start = probIdx
            self.p = self.E + self.alpha
            for word in doc.words:
                self.p = self.p * (self.lam * self.prob[probIdx]
                                   + (1 - self.lam) * (self.topicWordCountDMM[word] + self.beta) / (
                                   self.sumTopicWordCountDMM + Vbeta))  # F: 每个topic词的数量; prob shape = (sample_number, word_number)某个样本下一个词是word的概率;
                probIdx += 1

                if (np.sum(self.p) == 0):  # todo: self.p==0
                    dist = [1. / self.K] * self.K
                    topic = np.argmax(np.random.multinomial(1, dist))  # 以平均概率随机选择主题
                else:
                    dist = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
                    choices = range(len(dist))
                    topic = np.random.choice(choices, p=dist)

            self.E[topic] += 1
            self.F[topic] += doc.length

            for j in range(doc.length):
                word = doc.words[j]
                self.nw[word][topic] += 1

                subtopic = topic
                if ((self.lam * self.prob[prob_start])
                        > ((1 - self.lam) * (self.topicWordCountDMM[word][topic] + self.beta) / (
                        self.sumTopicWordCountDMM[topic] + Vbeta))):
                    self.topicWordCountLF[word][topic] += 1
                    self.sumTopicWordCountLF[topic] += 1
                else:
                    self.topicWordCountDMM[word][topic] += 1
                    self.sumTopicWordCountDMM[topic] += 1
                    subtopic += self.K
                doc.WordtopicAssignments[j] = subtopic
                prob_start += 1

            self.Z[i] = topic
            opt.topic_distribution[i] = dist

# if __name__=='__main__':
# 	loadpath='../data/news_train_title.p'
# 	dmm=DMMmodel(loadpath)
# 	dmm.est1()
# 	print(dmm.getTopicCoherence())
