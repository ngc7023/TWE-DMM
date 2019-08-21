# -*- co:ding: UTF-8 -*-
"""
 @Time    : 2019/8/19 9:26
 @Author  : Pangjy
 @File    : TWE4
 @Software: PyCharm
 Copy from TWE3, try to use english dataset
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
sys.path.append('/home/zliu/topic_modeling/TWE-DMM/')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from LEAM.main import *
from DMM.model2 import *
from TWE4.data_process import *
import cProfile
class TWE_Setting(object):
	def __init__(self):
		# LEAM
		self.GPUID = 0
		self.dataset = 'N20small'
		self.fix_emb = True         # Word embedding 初始化方式判断
		self.restore = False
		self.W_emb = None           # Word embedding初始化矩阵
		self.W_class_emb = None     # Label embedding 初始化矩阵
		self.maxlen = 30            # 序列最大长度
		self.n_words = None
		self.embed_size = 300       # embedding 维度
		self.lr = 1e-3              # 学习率
		self.batch_size = 8       # N20short-8 / TMNtitle-128
		self.max_epochs = 200       # default_200
		self.dropout = 0.5
		self.part_data = False
		self.portion = 1.0
		self.ifGammaUse = True
		self.setSampleNumber = 27728861

		self.save_path = ""
		self.log_path = ""
		self.topicwordemb_path = ""
		self.phifile = ""  # 词-主题分布文件phi
		self.thetafile = ""
		self.topNfile = "" # 每个主题topN词文件
		self.tagassignfile = ""  # 最后分派结果文件

		self.print_freq = 100
		self.valid_freq = 100
		self.num_class = 40         # 主题个数

		self.optimizer = 'Adam'
		self.clip_grad = 5
		self.class_penalty = 1.0
		self.ngram = 1.0
		self.H_dis = 64             # 全连接层单元个数，相当于输出维度？

		self.sample_number = None
		self.topic_distribution = None
		self.topic_emb = None
		self.gamma = None

		self.IterationforInitialization = 1
	def __iter__(self):
		for attr, value in self.__dict__.iteritems():
			yield attr, value

def main():
	np.set_printoptions(threshold=np.inf)
	# Prepare training and testing data
	opt = TWE_Setting()
	if opt.dataset == 'train_text':
		opt.loadpath = '../data/news_train_text.p'
		opt.embpath = "../data/train_text_emb.p"
		# opt.loadpath='/home/zliu/topic_modeling/TWE-DMM/data/news_train_text.p'
		# opt.embpath = "/home/zliu/topic_modeling/TWE-DMM/data/train_text_emb.p"
		# dmm = DMMmodel('../data/news_train_text.p')
		if opt.ifGammaUse:
			opt.save_path = "./save_largeset_gamma/"
			opt.log_path = "./log_largeset_gamma/"
			opt.topicwordemb_path = './re_largeset_gamma/topic-wordemb.p'
			opt.phifile = './re_largeset_gamma/phifile.txt'  # 词-主题分布文件phi
			opt.thetafile = './re_largeset_gamma/thetafile.txt'
			opt.topNfile = './re_largeset_gamma/topNfile.txt'  # 每个主题topN词文件
			opt.tagassignfile = './re_largeset_gamma/tassginfile.txt'  # 最后分派结果文件
		else:
			opt.save_path = "./save_largeset_without_gamma/"
			opt.log_path = "./log_largeset_without_gamma/"
			opt.topicwordemb_path = './re_largeset_without_gamma/topic-wordemb.p'
			opt.phifile = './re_largeset_without_gamma/phifile.txt'  # 词-主题分布文件phi
			opt.thetafile = './re_largeset_without_gamma/thetafile.txt'
			opt.topNfile = './re_largeset_without_gamma/topNfile.txt'  # 每个主题topN词文件
			opt.tagassignfile = './re_largeset_without_gamma/tassginfile.txt'  # 最后分派结果文件
	elif opt.dataset == 'train_title':
		opt.loadpath='/home/zliu/topic_modeling/TWE-DMM/data/news_train_title.p'
		opt.embpath = "/home/zliu/topic_modeling/TWE-DMM/data/train_title_emb.p"
		# dmm = DMMmodel('../data/news_train_title.p')
	elif opt.dataset=='classifydata':
		opt.loadpath = '../data/classifydata/classifydata_index.p'
		opt.embpath = "../data/classifydata/classifydata_emb.p"

		opt.save_path = "./save_classifydata/"
		opt.log_path = "./log_classifydata/"
		opt.topicwordemb_path = './re/topic-wordemb.p'
		opt.phifile = './re/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re/thetafile.txt'
		opt.topNfile = './re/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re/tassginfile.txt'  # 最后分派结果文件
	elif opt.dataset == 'N20short':
		opt.setSampleNumber = 24326 # number of subsequence
		opt.corpus_path = '../data/TACL-datasets/N20short.txt'
		opt.loadpath = '../data/TACL-datasets/N20short.p'
		opt.embpath = '../data/TACL-datasets/N20short_emb.p'

		opt.save_path = "./save/"
		opt.log_path = "./log/"

		opt.topicwordemb_path = './re/topic-wordemb.p'
		opt.phifile = './re/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re/thetafile.txt'
		opt.topNfile = './re/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re/tassginfile.txt'  # 最后分派结果文件

	elif opt.dataset == 'TMNfull':
		opt.setSampleNumber = 597933
		opt.corpus_path = '../data/TACL-datasets/TMNfull.txt'
		opt.loadpath = '../data/TACL-datasets/TMNfull.p'
		opt.embpath = '../data/TACL-datasets/TMNfull_emb.p'

		opt.save_path = "./save/"
		opt.log_path = "./log/"

		opt.topicwordemb_path = './re/topic-wordemb.p'
		opt.phifile = './re/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re/thetafile.txt'
		opt.topNfile = './re/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re/tassginfile.txt'  # 最后分派结果文件

	elif opt.dataset == 'TMNtitle':
		opt.setSampleNumber = 160234
		opt.corpus_path = '../data/TACL-datasets/TMNtitle.txt'
		opt.loadpath = '../data/TACL-datasets/TMNtitle.p'
		opt.embpath = '../data/TACL-datasets/TMNtitle_emb.p'

		opt.save_path = "./save_classifydata/"
		opt.log_path = "./log_classifydata/"

		opt.topicwordemb_path = './re_classifydata/topic-wordemb.p'
		opt.phifile = './re_classifydata/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re_classifydata/thetafile.txt'
		opt.topNfile = './re_classifydata/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re_classifydata/tassginfile.txt'  # 最后分派结果文件

	elif opt.dataset == 'N20small':
		opt.setSampleNumber = 35206
		opt.corpus_path = '../data/TACL-datasets/N20small.txt'
		opt.loadpath = '../data/TACL-datasets/N20small.p'
		opt.embpath = '../data/TACL-datasets/N20small_emb.p'

		opt.save_path = "./save_classifydata/"
		opt.log_path = "./log_classifydata/"

		opt.topicwordemb_path = './re_classifydata/topic-wordemb.p'
		opt.phifile = './re_classifydata/phifile.txt'  # 词-主题分布文件phi
		opt.thetafile = './re_classifydata/thetafile.txt'
		opt.topNfile = './re_classifydata/topNfile.txt'  # 每个主题topN词文件
		opt.tagassignfile = './re_classifydata/tassginfile.txt'  # 最后分派结果文件

	else:
		pass

	# Initialize DMM
	print("dataset:",opt.dataset)
	print("use gamma:",opt.ifGammaUse)
	dmm = DMMmodel(opt.loadpath,opt.num_class,opt)
	dmm.init_Z(dmm.Z)
	opt.n_words = dmm.dpre.words_count

	print("calculating topic coherence:")
	dmm._phi() # 计算Topic_Coherence初始值
	print("topic coherence:",dmm.getTopicCoherence())
	print("load data finished")
	print("docs_count", dmm.dpre.docs_count)
	print('total words: %d' % opt.n_words)
	print("batch_size:",opt.batch_size)
	print("topic number:",opt.num_class)
	print("save path:",opt.save_path)
	print("log path:",opt.log_path)
	print("dmm save path:",opt.tagassignfile)
	
	# Train TWE
	# Prepare Label
	x = cPickle.load(open(opt.loadpath, "rb"))
	train_lab, opt.sample_number = make_train_data(x[0],opt) # lab: 行列
	print('sample number: ', opt.sample_number)
	train_lab = np.array(train_lab, dtype='int32')
	del x

	opt.topic_distribution = np.zeros([dmm.dpre.docs_count,dmm.K],dtype='float32')

	for i in range(opt.IterationforInitialization):
		print(i)
		dmm.sampleSingleInitialIteration(opt)
	print("Calculating topic coherence")
	dmm._phi()
	print(dmm.getTopicCoherence())
	# print(opt.topic_distribution[0])

	print("Start Train_TWE")
	Train_TWE(opt,train_lab,dmm)

if __name__ == '__main__':
	main()
