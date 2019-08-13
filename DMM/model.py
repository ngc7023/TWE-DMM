# -*- coding: utf-8 -*-
import numpy as np
import random
import codecs
from collections import OrderedDict
import math
import _pickle as cPickle
import pandas as pd

class Document0(object):
	def __init__(self):
		self.words = []
		self.length = 0
		self.occurenceToIndexCount=[]

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
	def __init__(self, loadpath):
		x_data = cPickle.load(open(loadpath, "rb"))
		train=x_data[0]
		print(len(train))
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

			wordcount={}
			wordcount_list=[]
			for w in train[i]:
				if w in wordcount:
					wordcount[w]+=1
					wordcount_list.append(wordcount[w])
				else:
					wordcount[w]=1
					wordcount_list.append(wordcount[w])
			doc.occurenceToIndexCount=wordcount_list
			dpre.docs.append(doc)
		dpre.word2id=wordtoix
		print(dpre.docs_count)

		self.dpre = dpre  # 获取预处理参数（文档预处理实例）
		# 模型参数
		self.K = 40  # 主题个数
		self.beta = 0.01  # 每个主题下词的狄利克雷分布先验参数beta（超参数）
		self.alpha = 0.1  # 每个文档下主题的狄利克雷分布先验参数alpha（超参数）
		self.lam=0.5 # 决定参数
		self.iter_times = 2  # 最大迭代次数
		self.top_words_num = 20  # 每个主题特征词个数？？？？

		# self.prob=None # embedding预测单词的概率

		self.phifile = './re/phifile1.txt'  # 词-主题分布文件phi
		self.thetafile = './re/thetafile1.txt'
		self.topNfile = './re/topNfile1.txt'  # 每个主题topN词文件
		self.tassginfile = './re/tassginfile1.txt'  # 最后分派结果文件

		self.p = np.zeros(self.K)  # 概率向量double类型，存储采样的临时变量
		self.E = np.zeros(self.K,dtype = 'int')  # 每个主题的文档数
		self.ndsum = np.zeros(self.dpre.docs_count, dtype = 'int')  # 每个doc中词的总数
		self.F=np.zeros(self.K,dtype = 'int')  #每个topic词的数量
		self.nw = np.zeros((self.dpre.words_count, self.K), dtype = 'int')  # 词word在主题topic上的数量

		self.Z=np.zeros(self.dpre.docs_count,dtype = 'int')

		# 随机分配主题类型，为每个文档中的各个单词随机分配主题
		for x in range(self.dpre.docs_count):
			topic = random.randint(0, self.K - 1)  # 随机取一个主题
			self.Z[x] = topic  # 第x篇文档的主题为topic
			self.E[topic]+=1 # 每个主题的文档数
			self.ndsum[x] = self.dpre.docs[x].length
			for y in range(self.dpre.docs[x].length):
				self.nw[self.dpre.docs[x].words[y]][topic] += 1
				self.F[topic] += 1

		self.theta = np.array([0.0 for y in range(self.K)])  # 全数据集主题分布
		self.phi = np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])  # 主题-词分布

class DMMmodel1(object):
	def __init__(self, loadpath, K,opt):
		x_data = cPickle.load(open(loadpath, "rb"))
		train = x_data[0]
		train_lab = []
		wordtoix= x_data[2]
		# idxtoword = x_data[3]
		del x_data

		dpre = DataPreprocessing()
		dpre.docs_count = len(train)
		dpre.words_count = len(wordtoix)
		for i in range(len(train)):  # train[i]是某篇文档
			doc = Document()
			doc.words = train[i]
			doc.length = len(train[i])
			dpre.docs.append(doc)
		dpre.word2id = wordtoix
		del wordtoix
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

		self.phifile = opt.phifile  # 词-主题分布文件phi
		self.thetafile = opt.thetafile
		self.topNfile = opt.topNfile  # 每个主题topN词文件
		self.tagassignfile = opt.tagassignfile # 最后分派结果文件

		self.p = np.zeros(self.K)  # 概率向量double类型，存储采样的临时变量
		self.E = np.zeros(self.K, dtype='int')  # 每个主题的文档数
		self.ndsum = np.zeros(self.dpre.docs_count, dtype='int')  # 每个doc中词的总数
		self.F = np.zeros(self.K, dtype='int')  # 每个topic词的数量
		self.nw = np.zeros((self.dpre.words_count, self.K), dtype='int')  # 词word在主题topic上的数量

		if opt.restore:
			print("Restore Z")
			self.Z = self.Restore_Z()
		else:
			print("No restore, Initialize Z")
			self.Z = np.zeros(self.dpre.docs_count, dtype='int')
		self.theta = np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])  # 全数据集主题分布
		self.phi = np.array([[0.0 for y in range(self.dpre.words_count)] for x in range(self.K)])  # 主题-词分布

	def init_Z(self,Z):
		self.Z=Z
		if np.sum(self.Z)!=0: # 根据DMM的抽样结果恢复DMM模型
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
			self.p =self.p* (self.nw[word] + self.beta) / (self.F + Vbeta)
		p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
		topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题

		self.E[topic]+=1
		for  j in range(self.dpre.docs[i].length):
			word=self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
			self.nw[word][topic] += 1
			self.F[topic]+=1

		return topic
	#@profile
	def sampling1(self, i):
		# 换主题
		# print("before sampling:",self.E)
		topic = self.Z[i]  # 第i个文档的主题
		self.E[topic]-=1 # topic的文档数
		doc = self.dpre.docs[i]
		for word in doc.words:
			self.nw[word][topic] -= 1
		self.F[topic]-=doc.length
	
		Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 唯一单词总数

		# gibbs sample
		self.p=(self.E+self.alpha)
		for word in doc.words:
			self.p =self.p* (self.nw[word] + self.beta) / (self.F + Vbeta)
		if(np.sum(self.p)==0):
			p = [1./self.K]*self.K
			topic = np.argmax(np.random.multinomial(1, [1./self.K]*self.K))  # 以平均概率随机选择主题
		else:
			p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
			topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
		self.E[topic]+=1
		for word in doc.words:
			self.nw[word][topic] += 1
		self.F[topic] += doc.length		
		return topic,p
	#@profile
	def sampling2(self, i):
		# 换主题，减数字
		topic = self.Z[i]  # 第i个文档的主题
		self.E[topic] -= 1  # topic的文档数
		
		doc = self.dpre.docs[i]
		for word in doc.words:
			self.nw[word][topic] -= 1
		self.F[topic]-=doc.length
		Vbeta = self.dpre.words_count * self.beta  # dpre.words_count: 词汇表单词总数

		# gibbs sample
		probIdx = self.probIdx_start[i]
		self.p = (self.E + self.alpha)  # len(E) = K # doc-topic # 每个主题的文档数量+文档下主题的狄利克雷分布alpha
		for word in doc.words[1:]:  # 这篇文档的单词长度
			self.p = self.p * ((1 - self.lam) * (self.nw[word] + self.beta) / (self.F + Vbeta) + self.lam * self.prob[probIdx])  # F: 每个topic词的数量; prob shape = (sample_number, word_number)某个样本下一个词是word的概率;
			# nw[word]要换成
			probIdx += 1
		if (np.sum(self.p) == 0):
			p = [1./self.K]*self.K
			topic = np.argmax(np.random.multinomial(1, [1./self.K]*self.K))  # 以概率p的分布随机选择主题
		else:
			p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
			topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
		self.E[topic] += 1
		for word in doc.words:
			self.nw[word][topic] += 1
		self.F[topic] += doc.length
		return topic,p

	# 训练LDA模型
	def est(self):
		for x in range(self.iter_times):
			for i in range(self.dpre.docs_count):
				topic=self.sampling(i)
				self.Z[i]=topic
		self.save()

		# 训练DMM模型
	def est1(self,opt):
		self.probIdx = 0
		for i in range(self.dpre.docs_count):  # 总文档数
			topic,p = self.sampling1(i)
			self.Z[i] = topic
			opt.topic_distribution[i]=p

	def est2(self,opt):
		self.probIdx = 0
		for i in range(self.dpre.docs_count):  # 总文档数
			topic,p = self.sampling2(i)
			self.Z[i] = topic
			opt.topic_distribution[i]=p

	def est_parallel0(self,i):
		topic = self.sampling1(i)
		return topic

	def est_parallel(self,i):
		topic = self.sampling2(i)
		return topic

	def _theta(self):
		self.theta = self.E + self.alpha
		self.theta=np.array(self.theta / np.sum(self.theta))

	def _theta1(self):  # 文章-主题分布
		self.probIdx = 0
		for i in range(self.dpre.docs_count):
			Vbeta = self.dpre.words_count * self.beta
			theta = self.E + self.alpha
			for j in range(1,self.dpre.docs[i].length):  # default_(0,length)
				word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
				theta = theta * ((1 - self.lam) * (self.nw[word] + self.beta) / (self.F + Vbeta) + self.lam *self.prob[self.probIdx])
				self.probIdx += 1
			self.theta[i] = theta  # [[K],[K],....] doc_count

	def _theta2(self):  # 计算Topic_Distribution初始值
		self.probIdx = 0
		for i in range(self.dpre.docs_count):
			Vbeta = self.dpre.words_count * self.beta
			theta = self.E + self.alpha
			for j in range(1,self.dpre.docs[i].length):  # default_(0,length)
				word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
				theta = theta * (self.nw[word] + self.beta) / (self.F + Vbeta)
				self.probIdx += 1
			self.theta[i] = theta  # [[K],[K],....] doc_count

	def _phi(self):
		for i in range(self.K):
			self.phi[i] = (self.nw.T[i] + self.beta) / (self.F[i] + self.dpre.words_count * self.beta)
			self.phi[i]=np.array(self.phi[i]/np.sum(self.phi[i]))

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
				f.write('Topic' +str(x)+': ')
				twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
				twords.sort(key = lambda i: i[1], reverse = True)
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
		with codecs.open(self.tagassignfile, 'w') as f:
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

	# def GetTopicDistribution(self):
	# 	statistic = pd.value_counts(self.Z,normalize=False)
	# 	topic_distribution = np.zeros(self.K,dtype='float32')
	# 	for i in statistic.index:
	# 		topic_distribution[i] = statistic[i]
	# 	#print(topic_distribution)
	# 	topic_distribution[:] = [x / self.K for x in topic_distribution]
	# 	#topic_distribution = topic_distribution - np.max(topic_distribution)
	# 	#topic_distribution = np.exp(topic_distribution)/sum(np.exp(topic_distribution))
	#
	# 	print("topic_distribution:",topic_distribution)
		#sumi = 0
		#for i in topic_distribution:
			#sumi+=i
			#print(i)
		#print("sum:",sumi)
		#return topic_distribution
	def Restore_Z(self):
		data = pd.read_csv(self.tagassignfile, sep='\t', names=['topic'])
		Stored_Z = data['topic'].values.tolist()
		return Stored_Z
	def initTopicEmb(self,all_wordemb):
		self._phi() # 更新词-主题分布 K*word_count
		topic_emb = np.zeros([self.K,len(all_wordemb[0])],dtype='float32')
		for i in range(self.K):
			word_dis = self.phi[i]
			word_dis = word_dis[:,np.newaxis]
			topic_emb[i] = np.sum(np.multiply(word_dis,all_wordemb),axis=0)
		return topic_emb

	def sampleSingleInitialIteration(self):
		for i in range(self.dpre.docs_count):  # 总文档数
			topic = self.Z[i]
			doc = self.dpre.docs[i]

			self.E[topic] -= 1 # topic-doc
			# for word in doc.words:


if __name__=='__main__':
	loadpath='../data/news_train_title.p'
	dmm=DMMmodel1(loadpath)
	dmm.est1()
	print(dmm.getTopicCoherence())
