# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/31 13:30
 @Author  : Pangjy
 @File    : MDKLDA_model/LDA.py
 @Software: PyCharm
 Create this file based on LDA/model.py(QingTingTing)
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
		self.id2word = {}

class LDAmodel(object):
	def __init__(self, train,test,wordtoix,ixtoword,opt):
		# x_data = cPickle.load(open(loadpath, "rb"))
		# train, val, test = x_data[0],x_data[1],x_data[2]
		# train_lab, val_lab, test_lab = x_data[3],x_data[4], x_data[5],
		# wordtoix, ixtoword = x_data[6], x_data[7]
		# train+=val
		# train_lab += val
		# del x_data, val, val_lab
		# train作为有label的text，test作为无label的text

		# 由opt控制的变量
		self.K = opt.num_topic  # 主题个数
		self.corpus_path = opt.corpus_path
		self.phifile = opt.LDA_phifile  # 词-主题分布文件phi
		self.thetafile = opt.LDA_thetafile
		self.topNfile = opt.LDA_topNfile  # 每个主题topN词文件
		self.tagassignfile = opt.LDA_tagassignfile  # 最后分派结果文件

		dpre = DataPreprocessing()
		dpre.docs_count = len(train)+len(test)
		dpre.docs_withlabel_count = len(train)
		dpre.docs_withoutlabel_count = len(test)
		dpre.words_count = len(wordtoix)
		# print("total words:",dpre.words_count)
		# print(ixtoword[2227])
		dpre.word2id = wordtoix
		dpre.id2word = ixtoword
		del wordtoix, ixtoword

		# 为了用Gensim_getTopicCoherence()
		df = pd.read_csv(self.corpus_path, names=[''])
		list_df = df.values.tolist()
		self.text = [0] * dpre.docs_count
		count_text = 0
		for line in list_df:
			self.text[count_text] = line[0].split()
			count_text += 1
		self._dict = Dictionary(self.text)

		for i in range(len(train)):  # train[i]是某篇文档
			doc = Document()
			word_length = len(train[i])
			doc.words = train[i]
			# self.text[i] = list(map(lambda x: dpre.id2word[x], train[i]))
			# self.text[i] = train[i].apply(lambda x : self.ixtoword[x])
			doc.length = word_length
			dpre.docs.append(doc)

		for i in range(len(test)):  # train[i]是某篇文档
			doc = Document()
			word_length = len(test[i])
			doc.words = test[i]
			# self.text[i] = list(map(lambda x: dpre.id2word[x], train[i]))
			# self.text[i] = train[i].apply(lambda x : self.ixtoword[x])
			doc.length = word_length
			dpre.docs.append(doc)

		self.dpre = dpre  # 获取预处理参数（文档预处理实例）
		# 模型参数
		self.beta = opt.beta  # 每个主题下词的狄利克雷分布先验参数beta（超参数）
		self.alpha = opt.alpha  # 每个文档下主题的狄利克雷分布先验参数alpha（超参数）
		self.iter_times = 2000  # 最大迭代次数
		self.top_words_num = opt.top_words_num  # 每个主题特征词个数

		self.p = np.zeros(self.K)  # 概率向量double类型，存储采样的临时变量
		self.nw = np.zeros((self.dpre.words_count, self.K), dtype = 'int')  # 词word在主题topic上的分布（未归一化）
		self.nwsum = np.zeros(self.K, dtype = 'int')  # 每个topic的词的总数
		self.nd = np.zeros((self.dpre.docs_count, self.K), dtype = 'int')  # 每个doc中各个topic的词的总数
		self.ndsum = np.zeros(self.dpre.docs_count, dtype = 'int')  # 每个doc中词的总数
		self.Z = np.array(
			[[0 for y in range(self.dpre.docs[x].length)] for x in range(self.dpre.docs_count)])  # 每个文档中词的主题分配

		# 随机分配主题类型，为每个文档中的各个单词随机分配主题
		for x in range(self.dpre.docs_count):
			self.ndsum[x] = self.dpre.docs[x].length
			for y in range(self.dpre.docs[x].length):
				topic = random.randint(0, self.K - 1)  # 随机取一个主题
				self.Z[x][y] = topic  # 第x篇文档的第y个词的主题为topic
				self.nw[self.dpre.docs[x].words[y]][topic] += 1
				self.nd[x][topic] += 1
				self.nwsum[topic] += 1

		self.theta = np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])  # 文档-主题分布
		self.phi = np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])  # 主题-词分布

	def sampling(self, i, j):
		# 换主题
		topic = self.Z[i][j]  # 第i个文档第j个词的主题
		word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
		self.nw[word][topic] -= 1
		self.nd[i][topic] -= 1
		self.nwsum[topic] -= 1
		self.ndsum[i] -= 1  # 文档i的词总数-1

		Vbeta = self.dpre.words_count * self.beta
		Kalpha = self.K * self.alpha
		# gibbs sample 公式（LDA数学八卦公式29）
		self.p = (self.nw[word] + self.beta) / (self.nwsum + Vbeta) * (self.nd[i] + self.alpha) / (
				self.ndsum[i] + Kalpha)

		p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
		topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题

		self.nw[word][topic] += 1
		self.nwsum[topic] += 1
		self.nd[i][topic] += 1
		self.ndsum[i] += 1

		return topic

	# 训练LDA模型
	def est(self):
		for x in range(self.iter_times):
			print("LDA Iteration:", x)
			if((x+1)%10==0):
				self._phi()
				print("Gensim topic coherence:", self.Gensim_getTopicCoherence())
			for i in range(self.dpre.docs_count):
				for j in range(self.dpre.docs[i].length):
					topic = self.sampling(i, j)
					self.Z[i][j] = topic
		print("Gensim topic coherence:", self.Gensim_getTopicCoherence())
		self._theta()
		self._phi()
		self.save()

	def _theta(self):
		for i in range(self.dpre.docs_count):
			self.theta[i] = (self.nd[i] + self.alpha) / (self.ndsum[i] + self.K * self.alpha)

	def _phi(self):
		for i in range(self.K):
			self.phi[i] = (self.nw.T[i] + self.beta) / (self.nwsum[i] + self.dpre.words_count * self.beta)

	def save(self):
		with codecs.open(self.thetafile, 'w') as f:
			for x in range(self.dpre.docs_count):
				for y in range(self.K):
					f.write(str(self.theta[x][y]) + '\t')
				f.write('\n')

		# 保存phi词-主题分布
		with codecs.open(self.phifile, 'w') as f:
			for x in range(self.K):
				for y in range(self.dpre.words_count):
					f.write(str(self.phi[x][y]) + '\t')
				f.write('\n')

		# 保存每个主题topic的词
		with codecs.open(self.topNfile, 'w', 'utf-8') as f:
			self.top_words_num = min(self.top_words_num, self.dpre.words_count)
			for x in range(self.K):
				f.write('第' + str(x) + '个主题：' + '\n')
				twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
				twords.sort(key = lambda i: i[1], reverse = True)
				for y in range(self.top_words_num):
					word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
					f.write('\t' * 2 + word + '\t' + str(twords[y][1]) + '\n')
				# f.write('\t' * 2 + str(twords[y][0]) + '\t' + str(twords[y][1]) + '\n')

		# 保存最后退出时，文档的词分配的主题
		with codecs.open(self.tagassignfile, 'w') as f:
			for x in range(self.dpre.docs_count):
				for y in range(self.dpre.docs[x].length):
					f.write(str(self.dpre.docs[x].words[y]) + ':' + str(self.Z[x][y]) + '\t')
				f.write('\n')

	def getTopicCoherence(self):
		self.top_words_num = min(self.top_words_num, self.dpre.words_count)
		coherence = 0
		for x in range(self.K):
			twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
			twords.sort(key = lambda i: i[1], reverse = True)
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
					coherence += math.log((countij + 1) / (countj * counti) * self.dpre.docs_count)
		return coherence / self.K

	def Gensim_getTopicCoherence(self):
		invalid_topic = 0
		# get topn words
		topnwords = []
		self._phi()
		self.top_words_num = min(self.top_words_num, self.dpre.words_count)
		for x in range(self.K):
			twords = [(n, self.phi[x][n]) for n in range(1, self.dpre.words_count)]  # todo:topicN不够的时候怎么办
			twords.sort(key=lambda i: i[1], reverse=True)  # 根据phi[x][n]排序
			twords = twords[0:self.top_words_num]
			# self.phi[i] = (self.nw.T[i] + self.beta) / (self.nwsum[i] + self.dpre.words_count * self.beta)
			num = self.beta / (self.nwsum[x] + self.dpre.words_count * self.beta)
			num = float('%.19f' % num)
			# (1, 0.0001225790634959549) 表示N20small此处开始无有效的topNword
			if ((1, num) in twords):
				invalid_topic += 1
				twords = twords[:twords.index((1, num))]
			if (len(twords) > 1):
				list = [self.dpre.id2word[wordid] for (wordid, num) in twords[0:self.top_words_num]]
				topnwords.append(list)
		print("topnwords:",topnwords)
		print("topic number in setting: %d   invalid topic number: %d   final topic number: %d" % (
		self.K, invalid_topic, len(topnwords)))
		cm = CoherenceModel(topics=topnwords, texts=self.text, dictionary=self._dict,
                            window_size=5, coherence='c_uci', topn=self.top_words_num, processes=4)
		cm2 = CoherenceModel(topics=topnwords, texts=self.text, dictionary=self._dict,
                             window_size=5, coherence='c_npmi', topn=self.top_words_num, processes=4)
		return cm.get_coherence(), cm2.get_coherence()

