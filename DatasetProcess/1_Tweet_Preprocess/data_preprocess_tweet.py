"""
 @Time    : 2019/8/19 9:30
 @UpdateTime : 2019/9/22 15:12
 @Author  : Pangjy
 @File    : data_preprocess_tweet.py
 @Software: PyCharm
 Function process comes from TWE/preprocess_ch.py
 Input: google_list, stanford_list, full-corpus, emnlp_dict
 Output: tweet_filtered.txt, tweet_filteredlabel.txt (2019/9/5)
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

def getVocab():
    # model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    # google_list = list(model.vocab.keys())

    # cols = [0] * 301
    # cols[0] = "word"
    # for i in range(300):
    #     cols[i+1] = i+1
    #
    # vec_stanford = pd.read_csv('../data/glove.42B.300d.txt',names=cols,sep=' ',quoting=3) # quoting用来保留语料里的双引号
    # stanford_list = vec_stanford['word'].tolist()
    # file = open("stanford_list" + '.txt', 'w')
    # for i in range(len(stanford_list)):
    #     file.write(str(stanford_list[i]))  # write函数不能写int类型的参数，所以使用str()转化
    #     print(stanford_list[i])
    #     file.write('\n')
    # file.close()
    # print("stanford list finished")

    df = pd.read_csv("google_list.txt",names=['word2vec_word'])
    google_list = df['word2vec_word'].values.tolist()

    df2 = pd.read_csv("stanford_list.txt",names=['glove_word','nan'],sep=' ',quoting=3)
    stanford_list = df2['glove_word'].values.tolist()
    # print(len(stanford_list))
    return google_list,stanford_list

def demoji(text):
	emoji_pattern = re.compile("["
		u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
	                           "]+", flags=re.UNICODE)
	return(emoji_pattern.sub(r'', text))

def preprocess_corpus(google_list,stanford_list):
    full_corpus = pd.read_csv('full-corpus.csv',encoding='utf-8')
    del full_corpus['Sentiment']
    del full_corpus['TweetId']
    del full_corpus['TweetDate']
    df = full_corpus
    # 清除标点和数字等
    trantab_english = str.maketrans({key: None for key in string.punctuation})
    trantab_chinese = str.maketrans({key: None for key in hanzi.punctuation})
    trantab_number = str.maketrans({key: None for key in ['0','1','2','3','4','5','6','7','8','9','£','€']})

    wordlist_origin = [] #  词标准化
    wordlist_true = []
    emnlp_dict = pd.read_csv('emnlp_dict.txt',sep='	',names=['origin','true'])
    for index,row in emnlp_dict.iterrows():
        wordlist_origin.append(row['origin'])
        wordlist_true.append(row['true'])

    df_filtered = []
    df_filteredlabel = []
    frequency = collections.Counter([])
    for index, row in df.iterrows():
        Flag_removeline = False
        row_str = row['TweetText']
        row_label = row['Topic']

        row_str = demoji(row_str)
        row_str = row_str.translate(trantab_english)
        row_str = row_str.translate(trantab_chinese)
        row_str = row_str.translate(trantab_number)

        row_str = row_str.lower() # 小写
        row_str = row_str.replace("apple", "")
        row_str = row_str.replace("twitter", "")
        row_str = row_str.replace("microsoft", "")
        row_str = row_str.replace("google", "")
        row_str = row_str.replace("http", "") # 去掉链接

        stopWords = set(stopwords.words('english'))
        words = word_tokenize(row_str) # 分词
        line = []
        line_str = ""
        for word in words:
            if word in wordlist_origin:  # 单词标准化
                index = wordlist_origin.index(word)
                word = wordlist_true[index]
            word = re.sub(r'[^a-zA-Z]', '', word)  # 只保留英文
            if(word==''):
                Flag_removeline = True
                break
            else:
                line.append(word)
                line_str += word
        if(Flag_removeline):
            continue
        else:
            line2 = []
            for word in line:
                if len(word)>3 and word not in stopWords and word in google_list and word in stanford_list:
                        line2.append(word)
            if(len(line2)!=0):
                # frequency_line = collections.Counter(line2)
                # frequency = frequency + frequency_line
                df_filtered.append(line2)
                df_filteredlabel.append(row_label)
    # print(frequency)
    # print(len(frequency))
    Save_list(df_filtered, 'tweet_filtered_tmp')
    Save_list2(df_filteredlabel, 'tweet_filteredlabel_tmp')

    label_index = 0
    for line in df_filtered:
        line_str = ""
        for word in line:
            line_str = line_str+word+" "
            # if word in removedic:
            #     line.remove(word)
        if(detect(line_str)!='en'):
            print(len(df_filtered))
            print(line_str)
            df_filtered.remove(line)
            print(len(df_filtered))
            df_filteredlabel.pop(label_index)
            label_index+=1
        else:
            frequency_line = collections.Counter(line)
            frequency = frequency + frequency_line
            label_index+=1

    removedic = []
    for key in frequency:
        if(frequency[key]<3):
            removedic.append(key)

    label_index = 0
    for line in df_filtered:
        line_str = ""
        for word in line:
            if word in removedic:
                line.remove(word)
        if (len(line)==0):
            df_filtered.remove(line)
            df_filteredlabel.pop(label_index)
            label_index += 1
        else:
            label_index += 1

    print("number of doc:",len(df_filtered))
    return df_filtered,df_filteredlabel

def Save_list(list1, filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')  # 相当于Tab一下，换一个单元格
        file2.write('\n')  # 写完一行立马换行
    file2.close()

def Save_list2(list1, filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        file2.write(str(list1[i]))  # write函数不能写int类型的参数，所以使用str()转化
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
    # Tweet 数据清理（参考Nguyen2015）
    google_list,stanford_list= getVocab()
    corpus_filtered,corpus_filteredlabel = preprocess_corpus(google_list,stanford_list)
    Save_list(corpus_filtered, 'tweet_filtered')
    Save_list2(corpus_filteredlabel, 'tweet_filteredlabel')

