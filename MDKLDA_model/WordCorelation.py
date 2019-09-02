import pandas as pd

class TWE_Setting(object):
	def __init__(self):
		# LDA/DMM参数
		self.restore = False
		self.num_class = 40         # 主题个数
		self.sample_number = None
		self.topic_distribution = None
		self.topic_emb = None
		self.NN_gamma = None
		self.IterationforInitialization = 1
		self.top_words_num = 20
		self.beta = 0.1
		self.alpha = 1
		# LEAM
		self.GPUID = 0
		self.dataset = 'Tweet'
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
		self.ifNN_gammaUse = True
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

opt = TWE_Setting()

if opt.dataset == 'Tweet':
		opt.setSampleNumber = 16404
		opt.corpus_path = '../data/classifydata2/langdetect_tweet.txt'
		opt.loadpath = '../data/classifydata2/langdetect_tweet.p'
		opt.labelpath = '../data/classifydata2/langdetect_tweet_label.txt'
		if (opt.emb_type == 'word2vec'):
			opt.embpath = '../data/classifydata2/langdetect_tweet_word2vec_emb.p'
		elif (opt.emb_type == 'glove'):
			opt.embpath = '../data/classifydata2/langdetect_tweet_glove_emb.p'

corpus_path = '../data/classifydata2/langdetect_tweet.txt'
loadpath = '../data/classifydata2/langdetect_tweet.p'
# lda = LDA.LDAmodel(opt.loadpath, opt)
# from MDKLDA_model.MDKLDA import *
from MDKLDA_model.LDA import *
lda = LDAmodel(opt.loadpath,opt)
# statis = np.zeros((lda.dpre.words_count-1,lda.dpre.words_count-1),dtype='int')
# for line in lda.text:
# 	for word in line:
