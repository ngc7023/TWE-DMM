# -*- coding: utf-8 -*-
"""
Created on Mon May  1 01:53:46 2017

@author: DinghanShen
"""

import csv
import numpy as np
import os
import re
import _pickle as cPickle
import string
import pdb
from gensim.models import KeyedVectors
#from gensim.models import Word2Vec

def uniform_weight(nin, nout=None, scale=0.05):
    if nout == None:
        nout = nin
    W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))
    return W

print('Loading yelp data...')
loadpath = "./yelp_full.p"
x = cPickle.load(open(loadpath, "rb"),encoding='iso-8859-1')
train = x[0]
lab = x[3]
wordtoix = x[6]
print("len of train:", len(train))
print("len of lab:",len(lab))
print("len of word",len(wordtoix))
print("len of val",len(x[4]))
print("len of test",len(x[5]))

# wordtoix, ixtoword = x[6], x[7]
#
# print('Loading word2vec embedding...')
# # w2v = load_bin_vec('GoogleNews-vectors-negative300.bin', vocab)
# model = KeyedVectors.load_word2vec_format('./glove.840B.300d.txt', binary=False)
#
# print('Loading word embedding...')
# emb = []
# count = 0
# for i in range(len(ixtoword)):
#     if i == 0:
#         emb.append(np.zeros(300))
#     else:
#         if ixtoword[i] in model:
#             emb.append(model[ixtoword[i]])
#         else:
#             count += 1
#             temp = uniform_weight(300, 1, 0.01).reshape(300)
#             emb.append(temp)
#
# emb = np.array(emb, dtype='float32')
#
# cPickle.dump([emb], open("yelp_full_emb.p", "wb"))
#
# print('Glove vectors loaded!')
#
# pdb.set_trace()