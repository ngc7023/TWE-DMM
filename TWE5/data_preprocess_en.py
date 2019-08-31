"""
 @Time    : 2019/8/19 9:30
 @Author  : Pangjy
 @File    : TWE4/data_preprocess_en.py
 @Software: PyCharm
 Function process comes from TWE/preprocess_ch.py
 将txt语料转化为TWE-DMM使用的.p文件
"""

import pandas as pd
import re
from nltk import *
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections
import gensim

import codecs
import re
import _pickle as cPickle
import string
from zhon import hanzi
from langdetect import detect
from langdetect import detect_langs

from gensim.models.word2vec import Word2Vec
import _pickle

def precess(filename,encoding,outputfilename): # copy from TWE/preprocess_ch.py
	docno_list=[]
	doccontent_list=[]
	doctext_list=[]
	f=codecs.open(filename,'r')
	ix=0
	max_len=0
	for line in f:
		doctext_list.append(line)
		token_list=line.split()
		if max_len<len(token_list):
			max_len=len(token_list)
		if len(token_list)!=0:
			doccontent_list.append(token_list)
			docno_list.append(ix)
			ix+=1
	f.close()
	print('docnumber:',ix)
	print('max_len:',max_len)
	vocab={}
	for doc in doccontent_list:
		for w in doc:
			if w in vocab:
				vocab[w]+=1
			else:
				vocab[w]=1
	# vocab1={}                # modify by pjy
	# for w in vocab:
	# 	if vocab[w]>20:
	# 		vocab1[w]=vocab[w]

	print('len_vocab:',len(vocab))

	wordtoix = {}
	ixtoword = {}
	wordtoix['UNK'] = 0
	ixtoword[0] = 'UNK'
	ix = 1
	for w in vocab:             # modify by pjy
		wordtoix[w] = ix
		ixtoword[ix] = w
		ix += 1

	doccententindex = [];
	for doc in doccontent_list:
		docix_list = []
		for w in doc:
			if w in wordtoix:
				docix_list.append(wordtoix[w])
			else:
				docix_list.append(wordtoix['UNK'])
		doccententindex.append(docix_list)
	cPickle.dump([doccententindex,docno_list,wordtoix, ixtoword,doctext_list], open(outputfilename, "wb"))


if __name__=='__main__':
    filename = '../data/TACL-datasets/TMNfull.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/TMNfull.p'
    # precess(filename, encoding, outputfilename)

    # filename = '../data/TACL-datasets/TMNtitle.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/TMNtitle.p'
    # precess(filename, encoding, outputfilename)
    #
    # filename = '../data/TACL-datasets/N20small.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/N20small.p'
    # precess(filename, encoding, outputfilename)


    # filename = '../data/TACL-datasets/N20short.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/N20short.p'
    # precess(filename, encoding, outputfilename)

    # filename = '../data/TACL-datasets/N20.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/N20.p'
    # precess(filename, encoding, outputfilename)

    # filename = '../data/TACL-datasets/langdetect_tweet.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/langdetect_tweet.p'
    # precess(filename, encoding, outputfilename)

