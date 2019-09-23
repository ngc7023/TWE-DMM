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
import cProfile
class TopicModel_Setting(object):
	def __init__(self):
		# LDA/DMM参数
		self.restore = False
		self.num_topic = 10         # 主题个数
		self.num_class = 4          # Label个数
		self.top_words_num = 5
		self.beta = 0.1
		self.alpha = 1

		self.dataset = 'Tweet'
		self.proportion = 0.3
		self.emb_type = "glove"

		self.LDA_phifile = ""  # 词-主题分布文件phi
		self.LDA_thetafile = ""
		self.LDA_topNfile = "" # 每个主题topN词文件
		self.LDA_tagassignfile = ""  # 最后分派结果文件

		self.MDKLDA_phifile = ""  # 词-主题分布文件phi
		self.MDKLDA_thetafile = ""
		self.MDKLDA_topNfile = "" # 每个主题topN词文件
		self.MDKLDA_tagassignfile = ""  # 最后分派结果文件

	def __iter__(self):
		for attr, value in self.__dict__.iteritems():
			yield attr, value

def main():
	np.set_printoptions(threshold=np.inf)

	opt = TopicModel_Setting()
	if opt.dataset == 'Tweet':
		opt.corpus_path = '../DatasetProcess/1_Tweet_Preprocess/tweet_filtered.txt' # 文本始终使用这个
		opt.loadpath = '../DatasetProcess/2_Partition_Dataset_and_Generate_Embedding/outputdata/tweet_filtered'+str(opt.proportion)+'.p'
		opt.embpath = '../DatasetProcess/2_Partition_Dataset_and_Generate_Embedding/outputdata/tweet_filtered_emb.p'
		att_filename = '../DatasetProcess/3_Predict_Class_and_Get_Attention_Score/outputdata_fromLEAM/attention_score'+str(opt.proportion)+'.txt'
		predicted_label_file = '../DatasetProcess/3_Predict_Class_and_Get_Attention_Score/outputdata_fromLEAM/record_prob'+str(opt.proportion)+'.txt'

		opt.LDA_phifile = './re_LDA/re_tweet/'+str(opt.num_topic)+'/'+str(opt.proportion)+'/phifile.txt'  # 词-主题分布文件phi
		opt.LDA_thetafile = './re_LDA/re_tweet/'+str(opt.num_topic)+'/'+str(opt.proportion)+'/thetafile.txt'
		opt.LDA_topNfile = './re_LDA/re_tweet/'+str(opt.num_topic)+'/'+str(opt.proportion)+'/topNfile.txt'  # 每个主题topN词文件
		opt.LDA_tagassignfile = './re_LDA/re_tweet/'+str(opt.num_topic)+'/'+str(opt.proportion)+'/tassginfile.txt'  # 最后分派结果文件

		opt.MDKLDA_phifile = './re_MDKLDA/re_tweet/' +str(opt.num_topic)+'/'+ str(opt.proportion) + '/phifile.txt'  # 词-主题分布文件phi
		opt.MDKLDA_thetafile = './re_MDKLDA/re_tweet/' +str(opt.num_topic)+'/'+ str(opt.proportion) + '/thetafile.txt'
		opt.MDKLDA_topNfile = './re_MDKLDA/re_tweet/'+str(opt.num_topic)+'/' + str(opt.proportion) + '/topNfile.txt'  # 每个主题topN词文件
		opt.MDKLDA_tagassignfile = './re_MDKLDA/re_tweet/'+str(opt.num_topic)+'/' + str(opt.proportion) + '/tassginfile.txt'  # 最后分派结果文件

	elif opt.dataset == 'N20short':
		opt.corpus_path = '../DatasetProcess/TACL-datasets/N20short.txt'
		opt.loadpath = '../DatasetProcess/TACL-datasets/N20short'+str(opt.proportion)+'.p'
		if(opt.emb_type=='word2vec'):
			opt.embpath = '../data/TACL-datasets/N20short_word2vec_emb.p'
		elif(opt.emb_type=='glove'):
			opt.embpath = '../data/TACL-datasets/N20short_glove_emb.p'

		opt.LDA_phifile = './re_LDA/re_N20short/'+str(opt.num_topic)+'/'+str(opt.proportion)+'/phifile.txt'  # 词-主题分布文件phi
		opt.LDA_thetafile = './re_LDA/re_N20short/'+str(opt.num_topic)+'/'+str(opt.proportion)+'/thetafile.txt'
		opt.LDA_topNfile = './re_LDA/re_N20short/'+str(opt.num_topic)+'/'+str(opt.proportion)+'/topNfile.txt'  # 每个主题topN词文件
		opt.LDA_tagassignfile = './re_LDA/re_N20short/'+str(opt.num_topic)+'/'+str(opt.proportion)+'/tassginfile.txt'  # 最后分派结果文件

		opt.MDKLDA_phifile = './re_MDKLDA/re_N20short/' +str(opt.num_topic)+'/'+ str(opt.proportion) + '/phifile.txt'  # 词-主题分布文件phi
		opt.MDKLDA_thetafile = './re_MDKLDA/re_N20short/'+str(opt.num_topic)+'/' + str(opt.proportion) + '/thetafile.txt'
		opt.MDKLDA_topNfile = './re_MDKLDA/re_N20short/'+str(opt.num_topic)+'/' + str(opt.proportion) + '/topNfile.txt'  # 每个主题topN词文件
		opt.MDKLDA_tagassignfile = './re_MDKLDA/re_N20short/' +str(opt.num_topic)+'/'+ str(opt.proportion) + '/tassginfile.txt'  # 最后分派结果文件

	else:
		pass

	# Initilaize LDA
	print("topic number:",opt.num_topic)
	print("class number:",opt.num_class)
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
	print("train:",len(train))
	print("test",len(test))
	lda = LDAmodel(train, test, wordtoix, ixtoword, opt)
	lda.est()
	lda.save()
	print("lda save success")

	f = open(att_filename, 'r')
	Att_List = []
	stringList = f.readlines()
	for line in stringList:
		line_list = line.split()
		line_list = list(map(float, line_list))
		Att_List.append(line_list)
	mustsets_obj = MustSets()
	mustsets_obj.InitMustsetUseLEAM(train, train_lab, test, test_lab, wordtoix, ixtoword)
	mustsets_obj.ExtendMustSetsWithPredictLabelText(test, test_lab, ixtoword)
	mustsets_obj.InitializeRelatedWord(train, train_lab, test, test_lab, Att_List)

	try:
		mdklda = MDKLDAmodel(opt, lda, mustsets_obj)
		mdklda.test_lab = test_lab
		del test_lab
		mdklda.initializeFirstMarkovChainUsingExistingZ(lda.Z)
		mdklda.run()
		mdklda.save()

	except:
		print("Interrupt! MDKLDA model saved!")
		mdklda.save()

if __name__ == '__main__':
	main()
