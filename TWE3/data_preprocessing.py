import pandas as pd
import re
from nltk import *
# from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import collections
import gensim

from gensim.models.word2vec import Word2Vec
import _pickle

cols = [0] * 301
cols[0] = 'word'
for i in range(300):
    cols[i+1] = i

google_list = []
model = gensim.models.KeyedVectors.load_word2vec_format('/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/GoogleNews-vectors-negative300.bin', binary=True)
google_list = model.vocab
for key in google_list.keys:
    print(key)
    exit()

# google_list = []
# vec_google = pd.read_csv('/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/LFTM/TACL-datasets/TMNfull.Google300D.wordVectors',names=cols,sep=' ')
# for item in vec_google['word']:
#     google_list.append(item)
# print(len(google_list))

stanford_list = []
vec_stanford = pd.read_csv('/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/glove.42B.300d.txt',names=cols,sep=' ')
for item in vec_stanford['word']:
    stanford_list.append(item)

# model = Word2Vec.load('/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/LFTM/TACL-datasets/N20.Google300D.wordVectors')  # WORD_MODEL是我已经生成的模型
#
# print(model.wv.index2word())  # 获得所有的词汇
# for word in model.wv.index2word():
#     print(word, model[word])  # 获得词汇及其对应的向


full_corpus = pd.read_csv('/Users/wuyuanfujie/Code/Jupyter/TWE-DMM_preproccess/full-corpus.csv')
del full_corpus['Sentiment']
del full_corpus['TweetId']
del full_corpus['TweetDate']

emnlp_dict = pd.read_csv('/Users/wuyuanfujie/Code/PycharmCode/TWE-DMM/data/emnlp_dict.txt',sep='	',names=['origin','true'])
wordlist_origin = []
wordlist_true = []
for index,row in emnlp_dict.iterrows():
    wordlist_origin.append(row['origin'])
    wordlist_true.append(row['true'])

# list = ['a','a','a','b','c']
# dic = []
# f1= collections.Counter(list)
# for key in f1:
#     if(f1[key]<3):
#         dic.append(key)
# print(dic)
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
        if word!='' and word in wordlist_origin: # 单词标准化
            index = wordlist_origin.index(word)
            word = wordlist_true[index]
        if word!='' and len(word)>3 and word not in stopWords and word in stanford_list and word in google_list:
            line.append(word)
    if(len(line)!=0):
        frequency_line = collections.Counter(line)
        frequency = frequency + frequency_line
        str_filtered_stopwords.append(line)

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

# print(frequency)
print(len(str_filtered_stopwords))
# print(str_filtered_stopwords)
# print(len(str_filtered_stopwords))

