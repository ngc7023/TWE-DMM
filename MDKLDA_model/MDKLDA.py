# -*- coding: utf-8 -*-
"""
 @Time    : 2019/8/31 13:30
 @Author  : Pangjy
 @File    : MDKLDA/LDA.py
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

class MDKLDAmodel(object):
	def __init__(self, loadpath, opt, lda, mustsets_obj):
		self.dpre = lda.dpre # 定义在LDA里
		self.text = lda.text
		self._dict = lda._dict

		# mustset
		self.mustsets_obj = mustsets_obj
		self.MS = self.mustsets_obj.MS

		# 由opt控制的变量
		self.K = opt.num_topic  # 主题个数
		self.corpus_path = opt.corpus_path
		self.phifile = opt.phifile  # 词-主题分布文件phi
		self.thetafile = opt.thetafile
		self.topNfile = opt.topNfile  # 每个主题topN词文件
		self.tagassignfile = opt.tagassignfile  # 最后分派结果文件

		# 	The number of iterations for burn-in period.
		# todo 参数设定
		self.nBurnin = 0
		# 	The number of Gibbs sampling iterations.
		self.iter_times = 500  # 最大迭代次数
		# 	The length of interval to sample for calculating posterior distribution.
		self.sampleLag = 5
		self.lambdaForComputingGamma = 2000

		# Hyperparameters
		self.beta = opt.beta  # 每个主题下词的狄利克雷分布先验参数beta（超参数）
		self.alpha = opt.alpha  # 每个文档下主题的狄利克雷分布先验参数alpha（超参数）
		# 	// The coefficient randa for computing hyperparamter randaGamma
		# self.lambdaForComputingGamma = 2000
		# self.lambdaForComputingGamma = 200000000
		random.seed(0)
		# todo: randomseed

		self.tAlpha = self.K * self.alpha
		self.sBeta = self.MS * self.beta

		self.gamma = [0] * self.MS
		self.vGamma = np.zeros((self.K,self.MS), dtype='float64')
		for i in range(self.MS):
			size = self.mustsets_obj.len_mustsets[i]
			print(self.mustsets_obj.normalized_size)
			size_normalized = self.mustsets_obj.normalized_size[i]
			self.gamma[i] = [0] * size

			for j in range(size):
				self.gamma[i][j] = self.getGammaBasedOnMustsetSize(size_normalized)

		for t in range(self.K):
			for ms in range(self.MS):
				size = len(self.mustsets_obj.mustsets[ms])
				for i in range(size):
					self.vGamma[t][ms] += self.gamma[ms][i]

		self.top_words_num = opt.top_words_num  # 每个主题特征词个数

		self.ndt = np.zeros((self.dpre.docs_count, self.K), dtype = 'float')  # 每个doc中各个topic的词的总数
		self.ndsum = np.zeros(self.dpre.docs_count,dtype='float')

		self.nts = np.zeros((self.K,self.MS), dtype='float') # K * num(msset)
		self.ntsum = np.zeros(self.K,dtype='float') # K

		self.Z = np.array(
			[[0 for y in range(self.dpre.docs[x].length)] for x in range(self.dpre.docs_count)])  # 每个文档中词的主题分配
		self.Y = np.array(
			[[0 for y in range(self.dpre.docs[x].length)] for x in range(self.dpre.docs_count)])  # 每个文档中词的ms分配
		# // ntsw[k][ms][i]: number of ith word in the mset ms assigned to topic t and
		# // set ms, size T * MS * size of ms.
		self.ntsw = np.array([[[0 for y in range(len(self.mustsets_obj.mustsets[x]))] for x in range(self.MS)] for m in range(self.K)]) # K * num(msset) * sizeof(ms)
		# // ntsw[k][ms]: number of any word assigned to topic t and mset ms, size T *
		# // MS * size of ms.
		self.ntssum = np.zeros((self.K,self.MS), dtype='float') # K * num(msset)

		self.theta = np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])  # 文档-主题分布
		self.thetasum = np.array([[0.0 for y in range(self.K)] for x in range(self.dpre.docs_count)])
		self.phi = np.zeros((self.K,self.MS), dtype='float')  # Topic-mset distribution, size T * MS.
		self.phisum = np.zeros((self.K,self.MS), dtype='float')
		self.eta = np.array([[[0 for y in range(len(self.mustsets_obj.mustsets[x]))] for x in range(self.MS)] for m in range(self.K)]) # K * num(msset) * sizeof(ms)
		self.etasum = np.array([[[0 for y in range(len(self.mustsets_obj.mustsets[x]))] for x in range(self.MS)] for m in range(self.K)]) # K * num(msset) * sizeof(ms)
		self.omega = np.zeros((self.K,self.dpre.words_count),dtype='double')
		# 	Topic-word distribution, size T * V.
		# 	omega[t][w] = sum_{ms} phi[t][ms] *
		# 	eta[t][ms][w_i].
		# todo: words_count

		# 	Number of times to add the sum arrays, such as thetasum and phisum.
		self.numstats = 0
		self.test_lab = None

	def getGammaBasedOnMustsetSize(self, size):
		# return 1
		if(size==1):
			return 1
		else:
			# res = math.exp(math.log(self.lambdaForComputingGamma)-size)
			# res = math.exp(1850-size)
			# todo: set gamma
			# return math.exp(math.log(self.lambdaForComputingGamma)-size)
			return self.lambdaForComputingGamma/math.exp(size) # param.lambdaForComputingGamma = 2000

	def initializeFirstMarkovChainUsingExistingZ(self, Z2):
		print("initialize First MarkovChain Using Existing Z")
		for d in range(self.dpre.docs_withlabel_count):
			N = self.dpre.docs[d].length
			for n in range(N):
				word = self.dpre.docs[d].words[n]
				wordstr = self.dpre.id2word[word]

				topic = Z2[d][n]
				self.Z[d][n] = topic

				# Sample a must-set.
				mustsetList = self.mustsets_obj.getMustSetListGivenWordid(word)
				if(len(mustsetList)>1):
					ms = random.choice(mustsetList)
				else:
					ms = mustsetList[0]

				self.Y[d][n] = ms
				self.updateCount(d, topic, ms, word, wordstr, +1)

		for d in range(self.dpre.docs_withlabel_count, self.dpre.docs_count):
			N = self.dpre.docs[d].length
			for n in range(N):
				word = self.dpre.docs[d].words[n]
				wordstr = self.dpre.id2word[word]

				topic = Z2[d][n]
				self.Z[d][n] = topic

				# Sample a must-set.
				mustsetList = self.test_lab[d-self.dpre.docs_withlabel_count]
				set_idx = [i for i in range(len(mustsetList)) if mustsetList[i] == 1]
				ms = random.choice(set_idx)

				self.Y[d][n] = ms
				self.updateCount(d, topic, ms, word, wordstr, +1)

		# mustset类型统计
		# totalmustlist = []
		# for item in list(self.dpre.word2id.keys())[1:]:
		# 	wordstr = item
		# 	# wordstr = self.dpre.id2word[word]
		# 	# Sample a must-set.
		# 	mustsetList = mustset.getMustSetListGivenWordstr(wordstr)
		# 	totalmustlist.append(tuple(mustsetList))
		# keylist = set(totalmustlist)
		# for item in list(keylist):
		# 	print(item,totalmustlist.count(item))

	def updateCount(self, d, topic, ms, word, wordstr, flag):
		self.ndt[d][topic] += flag
		self.ndsum[d] += flag

		current_mustset = self.mustsets_obj.mustsets[ms]
		w1_i = current_mustset.index(word)

		self.ntsw[topic][ms][w1_i] += flag
		self.ntssum[topic][ms] += flag

		relatedwordset_w1 = self.mustsets_obj.relatedword[ms][w1_i]
		w2_idx_list = [pair[0] for pair in relatedwordset_w1]
		if(len(w2_idx_list)==0): # 有可能w1的beta为0
			count = flag * 1
			self.nts[topic][ms] += count
			self.ntsum[topic] += count
			if (self.nts[topic][ms] < 0):
				print("nts[topic][ms] < 0")
				print(d, topic, ms, word, wordstr, flag)
				exit()

		else:
			for w2_i in w2_idx_list:
				if (w1_i == w2_i):
					count = flag * 1
				else:
					count = flag * 0.5
				self.nts[topic][ms] += count
				self.ntsum[topic] += count
				if (self.nts[topic][ms] < 0):
					print("nts[topic][ms] < 0")
					print(d, topic, ms, word, wordstr, flag)
					exit()
		# for w2_i in range(len(current_mustset)):
		# 	if(w1_i==w2_i):
		# 		count = flag*1
		# 	else:
		# 		# count = flag * urn[ms][w1_i][w2_i];
		# 		count = flag*0.5
        #
		# 	self.nts[topic][ms] += count
		# 	self.ntsum[topic] += count
		# 	if(self.nts[topic][ms]<0):
		# 		print("nts[topic][ms] < 0")
		# 		print(d,topic,ms,word,wordstr,flag)
		# 		exit()

			# todo: This is different from M-LDA.
			# self.ntsw[topic][ms][w2_i] += count # 移到上面
			# self.ntssum[topic][ms] += count

	def run(self):
		self.runGibbsSampling()
		self.computePosteriorDistribution()
		print("Gensim topic coherence:", self.Gensim_getTopicCoherence())

	def runGibbsSampling(self):
		for i in range(self.iter_times):
			print("MDKLDA Iteration "+str(i))
			for d in range(self.dpre.docs_count):
				N = self.dpre.docs[d].length
				for n in range(N):
					self.sampleTopicAssignment(d, n)

			# print("updating posterior distribution")
			if (i >= self.nBurnin and self.sampleLag > 0 and i % self.sampleLag == 0):
				print("caocluting coherence")
				self.updatePosteriorDistribution()
				self.computePosteriorDistribution()
				print("Gensim topic coherence:", self.Gensim_getTopicCoherence())

			# # if(i%==0):
			# self.updatePosteriorDistribution()
			# self.computeDocumentTopicDistribution()
			# print("Gensim topic coherence:", self.Gensim_getTopicCoherence())


	def updatePosteriorDistribution(self):
		for d in range(self.dpre.docs_count):
			for t in range(self.K):
				self.thetasum[d][t] += (self.ndt[d][t]+self.alpha)/(self.ndsum[d]+self.tAlpha)

		for t in range(self.K):
			for ms in range(self.MS):
				self.phisum[t][ms] += (self.nts[t][ms]+self.beta)/(self.ntsum[t]+self.sBeta)

		for t in range(self.K):
			for ms in range(self.MS):
				size = len(self.mustsets_obj.mustsets[ms])
				for i in range(size):
					self.etasum[t][ms][i] += (self.ntsw[t][ms][i]+self.gamma[ms][i])/(self.ntssum[t][ms]+self.vGamma[t][ms])

		self.numstats+=1

	def sampleTopicAssignment(self,d,n):
		old_topic = self.Z[d][n]
		ms = self.Y[d][n]
		word = self.dpre.docs[d].words[n]
		wordstr = self.dpre.id2word[word]
		self.updateCount(d, old_topic, ms, word, wordstr, -1)

		if(d<self.dpre.docs_withlabel_count):
			mustsetList = self.mustsets_obj.getMustSetListGivenWordid(word)
		else:
			mustsetList = self.test_lab[d-self.dpre.docs_withlabel_count]
			mustsetList = [i for i in range(len(mustsetList)) if mustsetList[i] == 1]

		len_current_mustlist = len(mustsetList)
		# print(d,mustsetList)
		self.p = np.zeros(self.K*len_current_mustlist,dtype='float64')
		mustsets = self.mustsets_obj.mustsets
		for t in range(self.K):
			for si in range(len_current_mustlist):
				s = mustsetList[si] # si是mustsetList的索引 s是label的实际编号 e.g. mustsetList = [1, 2]
				mustset = mustsets[s]
				wordIndex = mustset.index(word)
				try:
					self.p[t * len(mustsetList)+si] = ((self.ndt[d][t] + self.alpha)/(self.ndsum[d] + self.tAlpha))\
				*((self.nts[t][s] + self.beta)/ (self.ntsum[t] + self.sBeta))\
				*((self.ntsw[t][s][wordIndex] + self.gamma[s][wordIndex])/(self.ntssum[t][s] + self.vGamma[t][s]))
				except:
					print(t,s,wordIndex)
				if(self.p[t * len(mustsetList)+si]<=0):
					print("The probability is xnegative!")
					print(self.ndt[d][t],self.ndsum[d],self.nts[t][s],self.ntsum[t],self.ntsw[t][s],self.gamma[s][wordIndex],self.ntssum[t][s],self.vGamma[t][s])
					exit()
		# Sample a topic and amust - set.
		pairIndex = self.sample(self.p,random.random())
		new_topic = pairIndex // len_current_mustlist # 向下取整
		new_ms = mustsetList[pairIndex % len_current_mustlist]
		self.Z[d][n] = new_topic
		self.Y[d][n] = new_ms

		self.updateCount(d, new_topic, new_ms, word, wordstr, +1)

	def computePosteriorDistribution(self):
		print("computing posterior distribution")
		self.computeDocumentTopicDistribution()
		self.computeTopicMustsetDistribution()
		self.computeTopicMustsetWordDistribution()
		self.computeTopicWordDistribution()

	def sample(self, p, randSeed):
		length = len(p)
		# Cumulative multinomial parameters
		cdf = [0] * length
		for x in range(length):
			cdf[x] = p[x]
		for x in range(length):
			cdf[x] += cdf[x-1]

		u = randSeed * cdf[length-1]
		for x in range(length):
			if(cdf[x]>u):
				return x

	# Document-topic distribution: theta[][].
	def computeDocumentTopicDistribution(self):
		if(self.sampleLag>0):
			for d in range(self.dpre.docs_count):
				for t in range(self.K):
					self.theta[d][t] = self.thetasum[d][t]/self.numstats
		else:
			for d in range(self.dpre.docs_count):
				for t in range(self.K):
					self.theta[d][t] = (self.ndt[d][t]+self.alpha)/(self.ndsum[d]+self.tAlpha)


	# Topic-mset distribution: phi[][].
	def computeTopicMustsetDistribution(self):
		if (self.sampleLag > 0):
			for t in range(self.K):
				for ms in range(self.MS):
					self.phi[t][ms] = self.phisum[t][ms] / self.numstats
		else:
			for t in range(self.K):
				for ms in range(self.MS):
					self.phi[t][ms] = (self.nts[t][ms]+self.beta)/(self.ntsum[t]+self.sBeta)

	# Topic-mset-word distribution: eta[][][].
	def computeTopicMustsetWordDistribution(self):
		if(self.sampleLag>0):
			for t in range(self.K):
				for ms in range(self.MS):
					size = len(self.mustsets_obj.mustsets[ms])
					for i in range(size):
						self.eta[t][ms][i] = self.etasum[t][ms][i]/self.numstats
		else:
			for t in range(self.K):
				for ms in range(self.MS):
					size = len(self.mustsets_obj.mustsets[ms])
					for i in range(size):
						self.eta[t][ms][i] = (self.ntsw[t][ms][i]+self.gamma[ms][i])/(self.ntssum[t][ms]+self.vGamma[t][ms])

	"""
     /**
	 * Topic-word distribution: omega[][][], not estimated from the model, but
	 * computed using phi and eta.
	 *
	 * omega[t][w] = sum_{s} phi[t][s] * eta[t][s][w_i]
	 */
    """

	def computeTopicWordDistribution(self):
		for t in range(self.K):
			for w in range(self.dpre.words_count):
				self.omega[t][w]= 0

		for t in range(self.K):
			for ms in range(self.MS):
				wordidList = self.mustsets_obj.mustsets[ms]
				for i in range(len(wordidList)):
					word = wordidList[i]
					prob = self.phi[t][ms] * self.eta[t][ms][i]
					self.omega[t][word] += prob

	# def sampling(self, i, j):
	# 	# 换主题
	# 	topic = self.Z[i][j]  # 第i个文档第j个词的主题
	# 	word = self.dpre.docs[i].words[j]  # 获取第i个文档第j个词的编号
	# 	self.nw[word][topic] -= 1
	# 	self.nd[i][topic] -= 1
	# 	self.nwsum[topic] -= 1
	# 	self.ndsum[i] -= 1  # 文档i的词总数-1
    #
	# 	Vbeta = self.dpre.words_count * self.beta
	# 	Kalpha = self.K * self.alpha
	# 	# gibbs sample 公式（LDA数学八卦公式29）
	# 	self.p = (self.nw[word] + self.beta) / (self.nwsum + Vbeta) * (self.nd[i] + self.alpha) / (
	# 			self.ndsum[i] + Kalpha)
    #
	# 	p = np.squeeze(np.array(self.p / np.sum(self.p)))  # squeeze:去掉数组形状中单维度条目
	# 	topic = np.argmax(np.random.multinomial(1, p))  # 以概率p的分布随机选择主题
    #
	# 	self.nw[word][topic] += 1
	# 	self.nwsum[topic] += 1
	# 	self.nd[i][topic] += 1
	# 	self.ndsum[i] += 1
    #
	# 	return topic



	# def _theta(self):
	# 	for i in range(self.dpre.docs_count):
	# 		self.theta[i] = (self.nd[i] + self.alpha) / (self.ndsum[i] + self.K * self.alpha)
    #
	# def _phi(self):
	# 	for i in range(self.K):
	# 		self.phi[i] = (self.nw.T[i] + self.beta) / (self.nwsum[i] + self.dpre.words_count * self.beta)
    #
	# def save(self):
	# 	with codecs.open(self.thetafile, 'w') as f:
	# 		for x in range(self.dpre.docs_count):
	# 			for y in range(self.K):
	# 				f.write(str(self.theta[x][y]) + '\t')
	# 			f.write('\n')
	# 	# 保存phi词-主题分布
	# 	with codecs.open(self.phifile, 'w') as f:
	# 		for x in range(self.K):
	# 			for y in range(self.dpre.words_count):
	# 				f.write(str(self.phi[x][y]) + '\t')
	# 			f.write('\n')
	# 	# 保存每个主题topic的词
	# 	with codecs.open(self.topNfile, 'w', 'utf-8') as f:
	# 		self.top_words_num = min(self.top_words_num, self.dpre.words_count)
	# 		for x in range(self.K):
	# 			f.write('第' + str(x) + '个主题：' + '\n')
	# 			twords = [(n, self.phi[x][n]) for n in range(self.dpre.words_count)]
	# 			twords.sort(key = lambda i: i[1], reverse = True)
	# 			for y in range(self.top_words_num):
	# 				word = OrderedDict({value: key for key, value in self.dpre.word2id.items()})[twords[y][0]]
	# 				f.write('\t' * 2 + word + '\t' + str(twords[y][1]) + '\n')
	# 			# f.write('\t' * 2 + str(twords[y][0]) + '\t' + str(twords[y][1]) + '\n')
	# 	# 保存最后退出时，文档的词分配的主题
	# 	with codecs.open(self.tagassignfile, 'w') as f:
	# 		for x in range(self.dpre.docs_count):
	# 			for y in range(self.dpre.docs[x].length):
	# 				f.write(str(self.dpre.docs[x].words[y]) + ':' + str(self.Z[x][y]) + '\t')

	def Gensim_getTopicCoherence(self):
		invalid_topic = 0
		# get topn words
		topnwords = []
		self.top_words_num = min(self.top_words_num, self.dpre.words_count)
		for x in range(self.K):
			twords = [(n, self.omega[x][n]) for n in range(2, self.dpre.words_count)]  # todo:topicN不够的时候怎么办
			twords.sort(key=lambda i: i[1], reverse=True)  # 根据phi[x][n]排序
			twords = twords[0:self.top_words_num]
			# self.phi[i] = (self.nw.T[i] + self.beta) / (self.nwsum[i] + self.dpre.words_count * self.beta)
			# num = self.beta / (self.nwsum[x] + self.dpre.words_count * self.beta)
			# num = float('%.19f' % num)
			# (1, 0.0001225790634959549) 表示N20small此处开始无有效的topNword
			# if ((1, num) in twords):
			# 	invalid_topic += 1
			# 	twords = twords[:twords.index((1, num))]
			if (len(twords) > 1):
				list = [self.dpre.id2word[wordid] for (wordid, num) in twords[0:self.top_words_num]]
				topnwords.append(list)
		print("top n words:", topnwords)
		print("topic number in setting: %d   invalid topic number: %d   final topic number: %d" % (
		self.K, invalid_topic, len(topnwords)))
		cm = CoherenceModel(topics=topnwords, texts=self.text, dictionary=self._dict,
                            window_size=10, coherence='c_uci', topn=self.top_words_num, processes=4)
		cm2 = CoherenceModel(topics=topnwords, texts=self.text, dictionary=self._dict,
                             window_size=10, coherence='c_npmi', topn=self.top_words_num, processes=4)
		return cm.get_coherence(), cm2.get_coherence()



