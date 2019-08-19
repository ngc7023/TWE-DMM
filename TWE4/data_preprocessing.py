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

from gensim.models.word2vec import Word2Vec
import _pickle

def getVocab():
    cols = [0] * 301
    cols[0] = 'word'
    for i in range(300):
        cols[i + 1] = i

    model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    google_list = list(model.vocab.keys())
    # print(google_list)
    print("Google list finished")
    # print(google_list.index('http'))
    # print(google_list.index('android'))

    vec_stanford = pd.read_csv('../data/glove.42B.300d.txt',names=cols,sep=' ')
    stanford_list = vec_stanford['word'].tolist()
    print("stanford list finished")
    # print(stanford_list.index('http'))
    # print(stanford_list.index('android'))
    return google_list,stanford_list

def preprocess_corpus(google_list,stanford_list):
    full_corpus = pd.read_csv('../data/full-corpus.csv')
    del full_corpus['Sentiment']
    del full_corpus['TweetId']
    del full_corpus['TweetDate']

    emnlp_dict = pd.read_csv('../data/emnlp_dict.txt',sep='	',names=['origin','true'])
    wordlist_origin = []
    wordlist_true = []
    for index,row in emnlp_dict.iterrows():
        wordlist_origin.append(row['origin'])
        wordlist_true.append(row['true'])
    frequency = collections.Counter([])

    df = full_corpus
    str_filtered_stopwords = []
    for index, row in df.iterrows():
        str = row['TweetText']
        str = str.lower() # 小写
        str = str.replace("apple", "")
        str = str.replace("twitter", "")
        str = str.replace("microsoft", "")
        str = str.replace("google", "")
        stopWords = set(stopwords.words('english'))
        words = word_tokenize(str) # 分词
        line = []
        for word in words:
            word = re.sub(r'[^a-zA-Z]', '', word)  # 只保留英文
            if(word==''):
                continue
            else:
                if word in wordlist_origin: # 单词标准化
                    index = wordlist_origin.index(word)
                    word = wordlist_true[index]
                if len(word)>3 and word not in stopWords and word in stanford_list and word in google_list:
                    line.append(word)
        if(len(line)!=0):
            frequency_line = collections.Counter(line)
            frequency = frequency + frequency_line
            str_filtered_stopwords.append(line)
    print(frequency)
    removedic = []
    for key in frequency:
        if(frequency[key]<3):
            removedic.append(key)

    for line in str_filtered_stopwords:
        for word in line:
            if word in removedic:
                line.remove(word)
            if(len(line)==0):
                str_filtered_stopwords.remove(line)
    return str_filtered_stopwords

def Save_list(list1, filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')  # 相当于Tab一下，换一个单元格
        file2.write('\n')  # 写完一行立马换行
    file2.close()

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
	vocab1={}
	for w in vocab:
		if vocab[w]>20:
			vocab1[w]=vocab[w]
	print('len_vocab:',len(vocab1))

	wordtoix = {}
	ixtoword = {}
	wordtoix['UNK'] = 0
	ixtoword[0] = 'UNK'
	ix = 1
	for w in vocab1:
		wordtoix[w] = ix
		ixtoword[ix] = w
		ix += 1

	doccententindex = []
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
    # google_list,stanford_list = getVocab()
    # corpus_filtered = preprocess_corpus(google_list,stanford_list)
    # Save_list(corpus_filtered, 'tweet_filtered')

    filename = '../data/corpus_title.txt'
    encoding = 'utf-8'
    outputfilename = '../data/news_train_title.p'
    precess(filename, encoding, outputfilename)