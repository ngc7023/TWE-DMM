# -*- coding: utf-8 -*-
import numpy as np
import random
import codecs
from collections import OrderedDict
import math
import _pickle as cPickle

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

class DMMmodel(object):
	def __init__(self, loadpath,K):
		x_data = cPickle.load(open(loadpath, "rb"))
		train=x_data[0]
		train_lab = []
		wordtoix, ixtoword = x_data[2], x_data[3]
		del x_data

		dpre=DataPreprocessing()
		dpre.docs_count=len(train)
		dpre.words_count=len(wordtoix)
		for i in range(len(train)):
			doc=Document()
			doc.words=train[i]
			doc.length=len(train[i])
			dpre.docs.append(doc)
		dpre.word2id=wordtoix

		self.dpre = dpre  # 获取预处理参数（文档预处理实例）
		# 模型参数
		self.K = K  # 主题个数
		self.beta = 0.1  # 每个主题下词的狄利克雷分布先验参数beta（超参数）
		self.alpha = 0.1  # 每个文档下主题的狄利克雷分布先验参数alpha（超参数）
		self.lam=0.5 # 决定参数
		self.iter_times = 100  # 最大迭代次数
		self.top_words_num = 20  # 每个主题特征词个数？？？？

		# topic word embedding 预测每个词的概率
		self.prob=[]
		self.probIdx=0

		self.phifile = './re/phifile.txt'  # 词-主题分布文件phi
		self.thetafile='./re/thetafile.txt'
		self.topNfile = './re/topNfile.txt'  # 每个主题topN词文件
		self.tassginfile = './re/tassginfile.txt'  # 最后分派结果文件

		self.p = np.zeros(self.K)  # 概率向量double类型，存储采样的临时变量
		self.E = np.zeros(self.K,dtype = 'int')  # 每个主题的文档数
		self.ndsum = np.zeros(self.dpre.docs_count, dtype = 'int')  # 每个doc中词的总数
		self.F=np.zeros(self.K,dtype = 'int')  #每个topic词的数量
		self.nw = np.zeros((self.dpre.words_count, self.K), dtype = 'int')  # 词word在主题topic上的数量

		self.Z=np.zeros(self.dpre.docs_count,dtype = 'int')

		self.theta = np.array([[0.0 for y in range(self.K)] for x  in range(self.dpre.docs_count)])  # 全数据集主题分布
		self.phi = np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])  # 主题-词分布

	def init_Z(self,Z):
		self.Z=Z
		if np.sum(self.Z)!=0:
			for x in range(len(self.Z)):
				topic = self.Z[x]
				self.E[topic] += 1  # 每个主题的文档数
				self.ndsum[x] = self.dpre.docs[x].length
				for y in range(self.dpre.docs[x].length):
					self.nw[self.dpre.docs[x].words[y]][topic] += 1
					self.F[topic] += 1
		else:
			# 随机分配主题类型，为每个文档中的各个单词随机分配主题
			for x in range(len(self.Z)):
				topic = random.randint(0, self.K - 1)  # 随机取一个主题
				self.Z[x] = topic  # 第x篇文档的主题为topic
				self.E[topic]+=1 # 每个主题的文档数
				self.ndsum[x] = self.dpre.docs[x].length
				for y in range(self.dpre.docs[x].length):
					self.nw[self.dpre.docs[x].words[y]][topic] += 1
					self.F[topic] += 1

	def sampling(self, i):
		# 换主题
		topic = self.Z[i]  # 第i个文档的主题
		self.E[topic]-=1 # topic的文档数
		for  j in range(self.dpre.docs[i].length):
			word=self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
			self.nw[word][topic] -= 1
			self.F[topic]-=1

		Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 唯一单词总数

		# gibbs sample
		self.p=(self.E+self.alpha)
		for  j in range(self.dpre.docs[i].length):
			word=self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
			self.p =self.p* ((1-self.lam)*(self.nw[word] + self.beta) / (self.F + Vbeta)+self.lam*self.prob[self.probIdx][word])
			self.probIdx+=1

		p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
		topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题

		self.E[topic]+=1
		for  j in range(self.dpre.docs[i].length):
			word=self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
			self.nw[word][topic] += 1
			self.F[topic]+=1

		return topic

	# 训练DMM模型
	def est(self):
		self.probIdx=0
		for i in range(self.dpre.docs_count):
			topic=self.sampling(i)
			self.Z[i]=topic


	def _theta(self):
		self.probIdx=0
		for i in range(self.dpre.docs_count):
			Vbeta = self.dpre.words_count * self.beta
			theta = self.E + self.alpha
			for j in range(self.dpre.docs[i].length):
				word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
				theta = theta * (
							(1 - self.lam) * (self.nw[word] + self.beta) / (self.F + Vbeta) + self.lam * self.prob[self.probIdx][word])
				self.probIdx+=1
			self.theta[i]=theta


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
					f.write(str(self.phi[x][y]) + ' ')
				f.write('\n')
		# 保存theta主题分布
		with codecs.open(self.thetafile, 'w') as f:
			for x in range(self.dpre.docs_count):
				for y in range(self.K):
					f.write(str(self.theta[x][y])+' ')
				f.write('\n')
			# 保存每个主题topic的topN词
			with codecs.open(self.topNfile, 'w', 'utf-8') as f:
				self.top_words_num = min(self.top_words_num, self.dpre.words_count)
				for x in range(self.K):
					f.write('Topic' + str(x) + ': ')
					twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
					twords.sort(key = lambda i: i[1], reverse = True)
					for y in range(self.top_words_num):
						word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
						f.write(str(twords[y][0]) + '(' + word + '_' + str(twords[y][1]) + ') ')
					f.write('\n')
		# 保存最后退出时，文档的主题
		with codecs.open(self.tassginfile, 'w') as f:
			for x in range(self.dpre.docs_count):
				f.write(str(self.Z[x]) + '\n ')

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
