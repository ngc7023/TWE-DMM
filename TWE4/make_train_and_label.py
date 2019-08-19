import pandas as pd
import re
from nltk import *
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections
import gensim

from gensim.models.word2vec import Word2Vec
import _pickle
"""
 @Time    : 2019/8/19 9:28
 @Author  : Pangjy
 @File    : TWE4/make_train_and_label.py
 @Software: PyCharm
 Try to make train and label(subsequences) from english corpus
"""

dataset = 'N20short'
if(dataset=='N20short'):
    txtpath = '../data/TACL-datasets/N20short.txt'
    embpath = '../data/TACL-datasets/N20.Google300D.wordVectors.txt'

def make_train_data1(full_text): # 2019-08-06
    label = []
    sample_num = 0
    for index,row in full_text.iterrows():
        words = word_tokenize(row['words'])
        for j in range(len(words)):
            label.append([index,j])
            sample_num += 1
    print(sample_num)
    return label,sample_num

full_text = pd.read_csv(txtpath,names=['words'])
make_train_data1(full_text)

cols = [0] * 301
cols[0] = 'word'
for i in range(300):
    cols[i+1] = i
emb = pd.read_csv(embpath,names=cols)

# print(full_text)



