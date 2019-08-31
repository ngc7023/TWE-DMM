# -*- co:ding: UTF-8 -*-
"""
 @Time    : 2019/8/31 13:45
 @Author  : Pangjy
 @File    : TWE5
 @Software: PyCharm
 Copy from TWE4, try to employ LDA & MDKLDA
"""

import os, sys
import _pickle as cPickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from math import floor
import multiprocessing
import timeit
import operator
# sys.path.append('/home/zliu/topic_modeling/TWE-DMM/')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from LEAM.main import *
from MDKLDA.LDA import *
from TWE5.data_process import *
import cProfile
class TWE_Setting(object):
	def __init__(self):
		# LDA/DMM参数
		self.restore = False
		self.num_class = 40         # 主题个数
		self.sample_number = None
		self.topic_distribution = None
		self.topic_emb = None
		self.gamma = None
		self.IterationforInitialization = 1

		# LEAM
		self.GPUID = 0
		self.dataset = 'tweet'
		self.emb_type = "word2vec"  # or glove
		self.fix_emb = True         # Word embedding 初始化方式判断
		self.W_emb = None           # Word embedding初始化矩阵
		self.W_class_emb = None     # Label embedding 初始化矩阵
		self.maxlen = 30            # 序列最大长度
		self.n_words = None
		self.embed_size = 300       # embedding 维度
		self.lr = 1e-3              # 学习率
		self.batch_size = 16        # N20short-16 / N20small-16/ Tweet-8
		self.max_epochs = 1         # default_200
		self.dropout = 0.5
		self.part_data = False
		self.portion = 1.0
		# LEAM - added
		self.ifGammaUse = True
		self.setSampleNumber = 0    # 记录subsequence的数量

		self.save_path = ""
		self.log_path = ""
		self.topicwordemb_path = ""
		self.phifile = ""  # 词-主题分布文件phi
		self.thetafile = ""
		self.topNfile = "" # 每个主题topN词文件
		self.tagassignfile = ""  # 最后分派结果文件

		self.print_freq = 100
		self.valid_freq = 100

		self.optimizer = 'Adam'
		self.clip_grad = 5
		self.class_penalty = 1.0
		self.ngram = 1.0
		self.H_dis = 64             # 全连接层单元个数，相当于输出维度？

	def __iter__(self):
		for attr, value in self.__dict__.iteritems():
			yield attr, value

def main():
	np.set_printoptions(threshold=np.inf)

	opt = TWE_Setting()

	if opt.dataset == 'N20short':
		opt.setSampleNumber = 24326 # number of subsequence
		opt.corpus_path = '../data/TACL-datasets/N20short.txt'
		opt.loadpath = '../data/TACL-datasets/N20short.p'
		if(opt.emb_type=='word2vec'):
			opt.embpath = '../data/TACL-datasets/N20short_word2vec_emb.p'
		elif(opt.emb_type=='glove'):
			opt.embpath = '../data/TACL-datasets/N20short_glove_emb.p'
		opt.save_path = "./save/save_N20short/"
		opt.log_path = "./log/log_N20short/"

		opt.topicwordemb_path = './re/re_N20short/topic-wordemb.p'
		opt.phifile = './re/re_N20short/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re/re_N20short/thetafile.txt'
		opt.topNfile = './re/re_N20short/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re/re_N20short/tassginfile.txt'  # 最后分派结果文件

	elif opt.dataset == 'N20small':
		opt.setSampleNumber = 35206
		opt.corpus_path = '../data/TACL-datasets/N20small.txt'
		opt.loadpath = '../data/TACL-datasets/N20small.p'
		if(opt.emb_type=='word2vec'):
			opt.embpath = '../data/TACL-datasets/N20small_word2vec_emb.p'
		elif (opt.emb_type == 'glove'):
			opt.embpath = '../data/TACL-datasets/N20small_glove_emb.p'

		opt.save_path = "./save/save_N20small"
		opt.log_path = "./log/log_N20small"

		opt.topicwordemb_path = './re/re_N20small/topic-wordemb.p'
		opt.phifile = './re/re_N20small/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re/re_N20small/thetafile.txt'
		opt.topNfile = './re/re_N20small/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re/re_N20small/tassginfile.txt'  # 最后分派结果文件

	elif opt.dataset == 'N20full':
		opt.setSampleNumber = 1944129
		opt.corpus_path = '../data/TACL-datasets/N20.txt'
		opt.loadpath = '../data/TACL-datasets/N20.p'
		if (opt.emb_type == 'word2vec'):
			opt.embpath = '../data/TACL-datasets/N20_emb.p'
		elif (opt.emb_type == 'glove'):
			opt.embpath = '../data/TACL-datasets/N20_glove_emb.p'
		opt.save_path = "./save/save_N20"
		opt.log_path = "./log/log_N20"

		opt.topicwordemb_path = './re/re_N20/topic-wordemb.p'
		opt.phifile = './re/re_N20/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re/re_N20/thetafile.txt'
		opt.topNfile = './re/re_N20/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re/re_N20/tassginfile.txt'  # 最后分派结果文件

	elif opt.dataset == 'TMNfull':
		opt.setSampleNumber = 597933
		opt.corpus_path = '../data/TACL-datasets/TMNfull.txt'
		opt.loadpath = '../data/TACL-datasets/TMNfull.p'
		if (opt.emb_type == 'word2vec'):
			opt.embpath = '../data/TACL-datasets/TMNfull_emb.p'
		elif (opt.emb_type == 'glove'):
			opt.embpath = '../data/TACL-datasets/TMNfull_glove_emb.p'

		opt.save_path = "./save/save_TMNfull"
		opt.log_path = "./log/log_TMNfull"

		opt.topicwordemb_path = './re/re_TMNfull/topic-wordemb.p'
		opt.phifile = './re/re_TMNfull/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re/re_TMNfull/thetafile.txt'
		opt.topNfile = './re/re_TMNfull/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re/re_TMNfull/tassginfile.txt'  # 最后分派结果文件

	elif opt.dataset == 'TMNtitle':
		opt.setSampleNumber = 160234
		opt.corpus_path = '../data/TACL-datasets/TMNtitle.txt'
		opt.loadpath = '../data/TACL-datasets/TMNtitle.p'
		if (opt.emb_type == 'word2vec'):
			opt.embpath = '../data/TACL-datasets/TMNtitle_emb.p'
		elif (opt.emb_type == 'glove'):
			opt.embpath = '../data/TACL-datasets/TMNtitle_glove_emb.p'

		opt.save_path = "./save_classifydata/"
		opt.log_path = "./log_classifydata/"

		opt.topicwordemb_path = './re_classifydata/topic-wordemb.p'
		opt.phifile = './re_classifydata/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re_classifydata/thetafile.txt'
		opt.topNfile = './re_classifydata/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re_classifydata/tassginfile.txt'  # 最后分派结果文件


	elif opt.dataset == 'Tweet':
		opt.setSampleNumber = 16404
		opt.corpus_path = '../data/TACL-datasets/langdetect_tweet.txt'
		opt.loadpath = '../data/TACL-datasets/langdetect_tweet.p'
		if (opt.emb_type == 'word2vec'):
			opt.embpath = '../data/TACL-datasets/langdetect_tweet_word2vec_emb.p'
		elif (opt.emb_type == 'glove'):
			opt.embpath = '../data/TACL-datasets/langdetect_tweet_glove_emb.p'

		opt.save_path = "./save/save_tweet/"
		opt.log_path = "./log/log_tweet/"

		opt.topicwordemb_path = './re/re_tweet/topic-wordemb.p'
		opt.phifile = './re/re_tweet/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re/re_tweet/thetafile.txt'
		opt.topNfile = './re/re_tweet/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re/re_tweet/tassginfile.txt'  # 最后分派结果文件

	else:
		pass

	# Initilaize LDA
	print("dataset:",opt.dataset)
	print("use gamma:",opt.ifGammaUse)
	print("embdding type",opt.emb_type)

	lda = LDAmodel(opt.loadpath, opt)
	# # Initialize DMM
	# print("dataset:",opt.dataset)
	# print("use gamma:",opt.ifGammaUse)
	# print("embdding type",opt.emb_type)
	# dmm = DMMmodel(opt.loadpath,opt.num_class,opt)
	# dmm.init_Z(dmm.Z)
	# opt.n_words = dmm.dpre.words_count
    #
	# dmm._phi() # 计算Topic_Coherence初始值
	# print("calculating topic coherence:")
	# print("Gensim topic coherence:",dmm.Gensim_getTopicCoherence())
    #
	# print("load data finished")
	# print("Topic number:",opt.num_class)
	# print("docs_count:", dmm.dpre.docs_count)
	# print('total words: %d' % opt.n_words)
	# print("batch_size:",opt.batch_size)
	# print("topic number:",opt.num_class)
	# print("save path:",opt.save_path)
	# print("log path:",opt.log_path)
	# print("dmm save path:",opt.tagassignfile)
    #
	# # Train TWE
	# # Prepare Label
	# x = cPickle.load(open(opt.loadpath, "rb"))
	# train_lab, opt.sample_number = make_train_data(x[0],opt) # lab: 行列
	# print('sample number: ', opt.sample_number)
	# train_lab = np.array(train_lab, dtype='int32')
	# del x
	# opt.topic_distribution = np.zeros([dmm.dpre.docs_count,dmm.K],dtype='float32')
    #
	# for i in range(opt.IterationforInitialization):
	# 	print(i)
	# 	dmm.sampleSingleInitialIteration(opt)
	# print("save dmm model\n")
	# dmm.save1(opt)
    #
	# dmm._phi()
	# print("calculating topic coherence:")
	# print("Gensim topic coherence(PMI NPMI):",dmm.Gensim_getTopicCoherence())
	# # print(dmm.E)
	# # print("calculating topic coherence:")
	# # print("topic coherence:",dmm.getTopicCoherence())
    #
	# print("Start Train_TWE")
	# Train_TWE(opt,train_lab,dmm)

if __name__ == '__main__':
	main()
