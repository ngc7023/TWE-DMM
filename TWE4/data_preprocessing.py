"""
 @Time    : 2019/8/19 9:30
 @Author  : Pangjy
 @File    : TWE4/data_preprocessing.py
 @Software: PyCharm
 Function process comes from TWE/preprocess_ch.py
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
    df = pd.read_csv("google_list.txt",names=['word2vec_word'])
    google_list = df['word2vec_word'].values.tolist()
    print("Google list finished")

    # cols = [0] * 301
    # cols[0] = 'word'
    # for i in range(300):
    #     cols[i + 1] = i
    # vec_stanford = pd.read_csv('../data/glove.42B.300d.txt',names=cols,sep=' ',quoting=3) # quoting用来保留语料里的双引号
    # stanford_list = vec_stanford['word'].tolist()
    # print("stanford list finished")
    return google_list

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

def preprocess_corpus(google_list):
    full_corpus = pd.read_csv('../data/full-corpus.csv',encoding='utf-8')
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
    emnlp_dict = pd.read_csv('../data/emnlp_dict.txt',sep='	',names=['origin','true'])
    for index,row in emnlp_dict.iterrows():
        wordlist_origin.append(row['origin'])
        wordlist_true.append(row['true'])

    df_filtered = []
    frequency = collections.Counter([])
    for index, row in df.iterrows():
        Flag_removeline = False
        row_str = row['TweetText']
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
                Flag_removeline = True # 确定本doc是英文的
                break
            else:
                line.append(word)
                line_str += word
        if(Flag_removeline):
            continue
        else:
            line2 = []
            for word in line:
                if len(word)>3 and word not in stopWords and word in google_list:
                    line2.append(word)
            if(len(line2)!=0):
                frequency_line = collections.Counter(line2)
                frequency = frequency + frequency_line
                df_filtered.append(line2)
    # print(frequency)
    # print(len(frequency))
    removedic = []
    for key in frequency:
        if(frequency[key]<3):
            removedic.append(key)

    for line in df_filtered:
        line_str = ""
        for word in line:
            line_str = line_str+word+" "
            if word in removedic:
                line.remove(word)
        if(len(line)==0 or detect(line_str)!='en'):
            df_filtered.remove(line)
            continue

    print("number of doc:",len(df_filtered))
    return df_filtered

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


def Detect_filtered():
    file = open("langdetect_tweet" + '.txt', 'w')
    df = pd.read_csv("tweet_filtered.txt",names='')
    count = 0
    for index,row in df.iterrows():
        res = str(detect_langs(row.name)[0])[:2]
        # print(res)
        if(res!="en"):
            print(row.name)
            count+=1
            continue
        else:
            file.write(str(row.name))
            file.write('\n')
            # print("writing")
    file.close()
    print(count)


if __name__=='__main__':
    # Tweet 数据清理，仍然转换为txt

    # Detect_filtered()
    # google_list= getVocab()
    # # google_list = []
    # corpus_filtered = preprocess_corpus(google_list)
    # Save_list(corpus_filtered, 'tweet_filtered')

    # 将原始语料转化为.p文件
    # filename = '../data/TACL-datasets/tweet_filtered.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/tweet_filtered.p'
    # precess(filename, encoding, outputfilename)

    # filename = '../data/TACL-datasets/TMNfull.txt'
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


    filename = '../data/TACL-datasets/N20short.txt'
    encoding = 'utf-8'
    outputfilename = '../data/TACL-datasets/N20short.p'
    precess(filename, encoding, outputfilename)

    # filename = '../data/TACL-datasets/N20.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/N20.p'
    # precess(filename, encoding, outputfilename)

    # filename = '../data/TACL-datasets/langdetect_tweet.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/TACL-datasets/langdetect_tweet.p'
    # precess(filename, encoding, outputfilename)