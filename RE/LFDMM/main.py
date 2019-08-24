# -*- co:ding: UTF-8 -*-
"""
 @Time    : 2019/8/12 9:32
 @Author  : Pangjy
 @File    : LFDMM/main.py
 @Software: PyCharm
 Create this file for calculate topic_coherence of the results of LFDMM, these results generated from directory LFTM
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

sys.path.append('../TWE-DMM/')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cProfile
from RE.LFDMM.DMMModel import *

class TWE_Setting(object):
    def __init__(self):
        # LEAM
        self.GPUID = 0
        self.dataset = 'TMNtitle'
        self.fix_emb = True  # Word embedding 初始化方式判断
        self.restore = True
        self.W_emb = None  # Word embedding初始化矩阵
        self.W_class_emb = None  # Label embedding 初始化矩阵
        self.maxlen = 300  # 序列最大长度
        self.n_words = None
        self.embed_size = 64  # embedding 维度
        self.lr = 1e-3  # 学习率
        self.batch_size = 128  # default_30
        self.max_epochs = 200  # default_200
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0

        # self.save_path = "./save_classifydata/"
        # self.log_path = "./log_classifydata/"
        # self.topicwordemb_path = './re/topic-wordemb.p'
        # self.phifile = './re/phifile1.txt'  # 词-主题分布文件phi
        # self.thetafile = './re/thetafile1.txt'
        # self.topNfile = './re/topNfile1.txt'  # 每个主题topN词文件
        # self.tagassignfile = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/RE/LFDMM/test/testLFDMM.topicAssignments'  # 最后分派结果文件

        self.print_freq = 100
        self.valid_freq = 100
        self.num_class = 7  # 主题个数

        self.optimizer = 'Adam'
        self.clip_grad = 5
        self.class_penalty = 1.0
        self.ngram = 1.0
        self.H_dis = 64  # 全连接层单元个数，相当于输出维度？

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
        opt.loadpath = '/home/zliu/topic_modeling/TWE-DMM/data/news_train_text.p'
        opt.embpath = "/home/zliu/topic_modeling/TWE-DMM/data/train_text_emb.p"
    # dmm = DMMmodel('../data/news_train_text.p')
    elif opt.dataset == 'train_title':
        opt.loadpath = '../data/news_train_title.p'
        opt.embpath = "/home/zliu/topic_modeling/TWE-DMM/data/train_title_emb.p"
    # dmm = DMMmodel('../data/news_train_title.p')
    elif opt.dataset == 'classifydata':
        opt.loadpath = '../data/classifydata/classifydata_index.p'
        opt.embpath = "../data/classifydata/classifydata_emb.p"
    # dmm = DMMmodel('../data/classifydata/classifydata_index.p')

    elif opt.dataset == 'Tweet':
        opt.setSampleNumber = 16404
        opt.corpus_path = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/TACL-datasets/langdetect_tweet.txt'
        opt.loadpath = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/TACL-datasets/langdetect_tweet.p'
        opt.embpath = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/TACL-datasets/langdetect_tweet_emb.p'

        opt.save_path = "./save/save_tweet/"
        opt.log_path = "./log/log_tweet/"

        opt.topicwordemb_path = './re/re_tweet/topic-wordemb.p'
        opt.phifile = './re/re_tweet/phifile.txt'  # 词-主题分布文件phi
        opt.thetafile = './re/re_tweet/thetafile.txt'
        opt.topNfile = './re/re_tweet/topNfile.txt'  # 每个主题topN词文件
        opt.tagassignfile = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/RE/LFDMM/test/testLFDMM.topicAssignments'  # 最后分派结果文件

    elif opt.dataset == 'TMNtitle':
        opt.setSampleNumber = 160234
        opt.corpus_path = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/TACL-datasets/TMNtitle.txt'
        opt.loadpath = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/TACL-datasets/TMNtitle.p'
        opt.embpath = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/TACL-datasets/TMNtitle_emb.p'

        opt.save_path = "./save_classifydata/"
        opt.log_path = "./log_classifydata/"

        opt.topicwordemb_path = './re_classifydata/topic-wordemb.p'
        opt.phifile = './re_classifydata/phifile.txt'  # 词-主题分布文件phi
        opt.thetafile = './re_classifydata/thetafile.txt'
        opt.topNfile = './re_classifydata/topNfile.txt'  # 每个主题topN词文件
        opt.tagassignfile = '/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/RE/LFDMM/test/testLFDMM.topicAssignments'  # 最后分派结果文件


    else:
        pass

    # Initialize DMM
    dmm = DMMmodel(opt.loadpath, opt.num_class, opt)
    dmm.init_Z(dmm.Z)
    opt.n_words = dmm.dpre.words_count

    if opt.dataset == 'train_text':
        dmm.sample_num = 27041529
    elif opt.dataset == 'classifydata':
        dmm.sample_num = 103648
        #dmm.sample_num = 107154
    dmm._phi()  # 计算Topic_Coherence初始值
    print("calculating topic coherence")
    print("topic coherence:", dmm.Gensim_getTopicCoherence())

    # print("load data finished")
    # print("docs_count", dmm.dpre.docs_count)
    # print('total words: %d' % opt.n_words)
    # print("batch_size:", opt.batch_size)
    # print("topic number:", opt.num_class)
    # print("save path:", opt.save_path)
    # print("log path:", opt.log_path)
    # print("dmm save path:", opt.tagassignfile)

if __name__ == '__main__':
    main()
