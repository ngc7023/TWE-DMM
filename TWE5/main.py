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
sys.path.append('/home/zliu/topic_modeling/TWE-DMM/')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from MDKLDA_model.LDA import *
from MDKLDA_model.MDKLDA import *
from MDKLDA_model.MustSet import *
from TWE5.data_process import *
import cProfile
class TopicModel_Setting(object):
	def __init__(self):
		# LDA/DMM参数
		self.restore = False
		self.num_topic = 4         # 主题个数
		self.num_class = 4          # Label个数
		self.top_words_num = 20
		self.beta = 0.1
		self.alpha = 1

		self.dataset = 'Tweet'
		self.emb_type = "word2vec"  # or glove

		self.save_path = ""
		self.log_path = ""
		self.topicwordemb_path = ""
		self.phifile = ""  # 词-主题分布文件phi
		self.thetafile = ""
		self.topNfile = "" # 每个主题topN词文件
		self.tagassignfile = ""  # 最后分派结果文件


	def __iter__(self):
		for attr, value in self.__dict__.iteritems():
			yield attr, value

def main():
	np.set_printoptions(threshold=np.inf)

	opt = TopicModel_Setting()

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


	elif opt.dataset == 'Tweet':
		opt.corpus_path = '../data/classifydata2/tweet_filtered.txt'
		opt.loadpath = '../data/classifydata2/tweet_filtered0.7.p'
		opt.embpath = '../data/classifydata2/tweet_filtered_emb.p'
		att_filename = '../LEAM_data/attention_score.txt'
		predicted_label_file = '../data/classifydata2/tweetlabel0.7.txt'

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
	print("dataset:", opt.dataset)
	x_data = cPickle.load(open(opt.loadpath, "rb"))
	train, val, test = x_data[0], x_data[1], x_data[2]
	train_lab, val_lab, test_lab_true = x_data[3], x_data[4], x_data[5],
	wordtoix, ixtoword = x_data[6], x_data[7]
	train += val
	train_lab += val_lab
	test_lab = []

	df = pd.read_csv(predicted_label_file,sep=' ',names=['0','1','2','3'])
	for ix,row in df.iterrows():
		tmp_lab = [0] * opt.num_class
		for i in range(opt.num_class):
			if(row[i]!=0):
				tmp_lab[i] = 1
		test_lab.append(tmp_lab)

	lda = LDAmodel(train, test, wordtoix, ixtoword, opt)
	lda.est()

	f = open(att_filename, 'r')
	Att_List = []
	stringList = f.readlines()
	for line in stringList:
		line_list = line.split()
		line_list = list(map(float, line_list))
		Att_List.append(line_list)
	mustsets_obj = MustSets()
	mustsets_obj.InitMustsetUseLEAM(train, train_lab, test, test_lab, wordtoix, ixtoword)
	# print(len(mustsets_obj.mustsets[0]),len(mustsets_obj.mustsets[1]),len(mustsets_obj.mustsets[2]),len(mustsets_obj.mustsets[3]))
	mustsets_obj.ExtendMustSetsWithPredictLabelText(test, test_lab,ixtoword)
	mustsets_obj.InitializeRelatedWord(train, train_lab, test, test_lab, Att_List)

	# print(len(mustsets_obj.mustsets[0]),len(mustsets_obj.mustsets[1]),len(mustsets_obj.mustsets[2]),len(mustsets_obj.mustsets[3]))

	# idx = wordtoix['crack']
	# print(mustsets_obj.wordidToMustsetListMap.get(idx))
	# exit()

	# opt.labelpath = '../data/classifydata2/langdetect_tweet_label.txt'
	# mustsets_obj.InitMustSets(lda.dpre.word2id,lda.text,opt.labelpath)
	#
	# opt.n_words = lda.dpre.words_count

	mdklda = MDKLDAmodel(opt.loadpath, opt, lda, mustsets_obj)
	mdklda.test_lab = test_lab
	del test_lab
	mdklda.initializeFirstMarkovChainUsingExistingZ(lda.Z)
	# exit()
	mdklda.run()

if __name__ == '__main__':
	main()
