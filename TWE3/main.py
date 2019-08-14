# -*- co:ding: UTF-8 -*-
"""
 @Time    : 2019/8/13 14:32
 @Author  : Pangjy
 @File    : TWE3
 @Software: PyCharm
 Copy from TWE2, this file use DMM/model2
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
from TWE3.data_process import *
import cProfile
class TWE_Setting(object):
	def __init__(self):
		# LEAM
		self.GPUID = 0
		self.dataset = 'classifydata'
		self.fix_emb = True         # Word embedding 初始化方式判断
		self.restore = False
		self.W_emb = None           # Word embedding初始化矩阵
		self.W_class_emb = None     # Label embedding 初始化矩阵
		self.maxlen = 30            # 序列最大长度
		self.n_words = None
		self.embed_size = 64        # embedding 维度
		self.lr = 1e-3              # 学习率
		self.batch_size = 128       # default_30
		self.max_epochs = 200       # default_200
		self.dropout = 0.5
		self.part_data = False
		self.portion = 1.0
		
		self.save_path = "./save_classifydata/"
		self.log_path = "./log_classifydata/"
		self.topicwordemb_path = './re/topic-wordemb1.p'
		self.phifile = './re/phifile1.txt'  # 词-主题分布文件phi
		self.thetafile = './re/thetafile1.txt'
		self.topNfile = './re/topNfile1.txt'  # 每个主题topN词文件
		self.tagassignfile = './re/tassginfile1.txt'  # 最后分派结果文件
		
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
	def __iter__(self):
		for attr, value in self.__dict__.iteritems():
			yield attr, value

def main():
	np.set_printoptions(threshold=np.inf)
	# Prepare training and testing data
	opt = TWE_Setting()
	if opt.dataset == 'train_text':
		opt.loadpath='/home/zliu/topic_modeling/TWE-DMM/data/news_train_text.p'
		opt.embpath = "/home/zliu/topic_modeling/TWE-DMM/data/train_text_emb.p"
		# dmm = DMMmodel('../data/news_train_text.p')
	elif opt.dataset == 'train_title':
		opt.loadpath='/home/zliu/topic_modeling/TWE-DMM/data/news_train_title.p'
		opt.embpath = "/home/zliu/topic_modeling/TWE-DMM/data/train_title_emb.p"
		# dmm = DMMmodel('../data/news_train_title.p')
	elif opt.dataset=='classifydata':
		opt.loadpath = '../data/classifydata/classifydata_index.p'
		opt.embpath = "../data/classifydata/classifydata_emb.p"
		# opt.loadpath = '/home/zliu/topic_modeling/TWE-DMM/data/classifydata/classifydata_index.p'
		# opt.embpath = "/home/zliu/topic_modeling/TWE-DMM/data/classifydata/classifydata_emb.p"
		# dmm = DMMmodel('../data/classifydata/classifydata_index.p')
	else:
		pass

	# Initialize DMM
	dmm = DMMmodel(opt.loadpath,opt.num_class,opt)
	dmm.init_Z(dmm.Z)
	opt.n_words = dmm.dpre.words_count

	if opt.dataset == 'train_text':
		dmm.sample_num = 27041529
	elif opt.dataset=='classifydata':
		dmm.sample_num = 103648
		#dmm.sample_num = 107154
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
	# x = cPickle.load(open(opt.loadpath, "rb"))
	# train_lab, opt.sample_number = make_train_data1(x[0],dmm) # lab: 行列 # ProbIdx：文本i的sample的起止值
	# print('sample number: ', opt.sample_number)
	# train_lab = np.array(train_lab, dtype='int32')
	# del x
	opt.topic_distribution = np.zeros([dmm.dpre.docs_count,dmm.K],dtype='float32')
	#opt.gamma = np.zeros([opt.batch_size,opt.num_class],dtype='float32')
	# 训练DMM，获取最初的Topic_Distribution，无prob
	# dmm.est1(opt)
	for i in range(200):
		print(i)
		dmm.sampleSingleInitialIteration()
	dmm._phi()
	print("topic coherence:",dmm.getTopicCoherence())
	print(opt.topic_distribution[0])

	#print("Start Train_TWE")
	#Train_TWE(opt,train_lab,dmm)

if __name__ == '__main__':
	main()
