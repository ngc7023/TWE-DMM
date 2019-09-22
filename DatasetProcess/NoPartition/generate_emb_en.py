"""
 @Time    : 2019/8/19 22:30
 @Author  : Pangjy
 @File    : TWE4/generate_emb_en.py
 @Software: PyCharm
 Copy from TWE/generate_emb_zh.py
 根据data_preprocess_en.py生成的.p文件，生成对应的emb.p文件，有两种embedding可以使用
"""

import gensim
import numpy as np
import codecs
import _pickle as cPickle

embedding =  "word2vec"
embedding =  "glove"

if(embedding=="word2vec"):
	filename = '../data/GoogleNews-vectors-negative300.bin'
	model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)

elif(embedding=="glove"):
	filename = '../data/glove.42B.300d.word2vecformat.txt'
	model = gensim.models.KeyedVectors.load_word2vec_format(filename,binary=False)


# filename = '../data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'

# 产出用于TWE-DMM的emb.p文件
def generate_emb(datafile,outputfile):
	loadpath = datafile
	x = cPickle.load(open(loadpath, "rb"))
	# train= x[0]
	# train_lab = x[1]
	wordtoix, ixtoword = x[2], x[3]
	vector_size = model.vector_size
	print(vector_size)
	embedding_vectors = np.random.uniform(-0.001, 0.001, (len(wordtoix), vector_size))
	glove_vocab = list(model.vocab.keys())
	count = 0
	mis_count = 0
	for word in wordtoix.keys():
		idx = wordtoix.get(word)
		if(word!='UNK'):
			if word in glove_vocab:
				embedding_vectors[idx] = model.wv[word]
				count += 1
			else:
				emb_str = ' '.join(str(i) for i in embedding_vectors[idx])
				mis_count += 1

	if(embedding=="word2vec"):
		print("num of vocab in word2vec: {}".format(count))
		print("num of vocab not in word2vec: {}".format(mis_count))
	elif(embedding=="glove"):
		print("num of vocab in glove: {}".format(count))
		print("num of vocab not in glove: {}".format(mis_count))
	cPickle.dump([embedding_vectors],open(outputfile, 'wb'))
	print("vocabulary length:",len(ixtoword))
	print(np.shape(embedding_vectors))

# 产出用于LFTM的wordVector.txt文件
def generate_emb2(datafile,outputfile):
	loadpath = datafile
	x = cPickle.load(open(loadpath, "rb"))
	# train= x[0]
	# train_lab = x[1]
	wordtoix, ixtoword = x[2], x[3]
	vector_size = model.vector_size
	print(vector_size)
	embedding_vectors = np.random.uniform(-0.001, 0.001, (len(wordtoix), vector_size)) # tweet数据已经根据google_list和stanford_list筛选过了，都有对应的emb，不会随机
	glove_vocab = list(model.vocab.keys())
	count = 0
	mis_count = 0
	file = open(outputfile, 'w')
	# print(wordtoix.get('close'))
	# exit()
	for word in wordtoix.keys():
		idx = wordtoix.get(word)
		if(word!='UNK'):
			if word in glove_vocab:
				embedding_vectors[idx] = model.wv[word]
				emb_str = ' '.join(str(format(i,'.6f')) for i in embedding_vectors[idx])
				file.write(str(word)+" "+emb_str)
				file.write('\n')
				count += 1
			else:
				emb_str = ' '.join(str(format(i,'.6f')) for i in embedding_vectors[idx])
				file.write(str(word)+" "+emb_str)
				file.write('\n')
				mis_count += 1

	print("num of vocab in word2vec: {}".format(count))
	print("num of vocab not in word2vec: {}".format(mis_count))
	# print("num of vocab in glove: {}".format(count))
	# print("num of vocab not in glove: {}".format(mis_count))
	file.close()
	# cPickle.dump([embedding_vectors],open(outputfile, 'wb'))
	print(len(ixtoword))
	print(np.shape(embedding_vectors))

def emb_LFTM(embfile,vocabfile,outputfile):
	x = cPickle.load(open(vocabfile, "rb"))
	ixtoword=x[3]
	del x
	x = cPickle.load(open(embfile, "rb"))
	emb=x[0]
	print(np.shape(x))
	del x
	print(len(ixtoword))
	print(np.shape(emb))
	ouf=codecs.open(outputfile,'w')
	for i in range(len(emb)):
		line=ixtoword[i]+' '
		for j in range(len(emb[i])):
			line+=str(emb[i][j])+' '
		# print(line)
		ouf.write(line+'\n')
	ouf.close()



if __name__=='__main__':
	datafile = '../data/TACL-datasets/langdetect_tweet.p'
	if(embedding=='word2vec'):
		outputfile = '../data/TACL-datasets/langdetect_tweet_Google300D.wordVectors'
	elif(embedding=='glove'):
		outputfile = '../data/TACL-datasets/langdetect_tweet_Stanford300D.wordVectors'
	generate_emb2(datafile, outputfile)

	# datafile = '../data/TACL-datasets/TMNtitle.p'
	# outputfile = '../data/TACL-datasets/TMNtitle_glove_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile = '../data/TACL-datasets/TMNfull.p'
	# outputfile = '../data/TACL-datasets/TMNfull_glove_emb.p'
	# generate_emb(datafile, outputfile)
    #
	# datafile = '../data/TACL-datasets/N20small.p'
	# outputfile = '../data/TACL-datasets/N20small_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile = '../data/TACL-datasets/N20small.p'
	# outputfile = '../data/TACL-datasets/N20small_glove_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile = '../data/TACL-datasets/N20.p'
	# outputfile = '../data/TACL-datasets/N20_glove_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile = '../data/TACL-datasets/langdetect_tweet.p'
	# outputfile = '../data/TACL-datasets/langdetect_tweet_word2vec_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile='../data/TACL-datasets/N20short.p'
	# outputfile='../data/TACL-datasets/N20short_glove_emb.p'
	# generate_emb(datafile,outputfile)

	# datafile='../data/news_train_text.p'
	# outputfile='./data/train_text_emb.p'
	# generate_emb(datafile,outputfile)

	# datafile = '../data/news_train_title.p'
	# outputfile = './data/train_title_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile='../data/train_title_emb.p'
	# vocabfile='../data/news_train_title.p'
	# outputfile='../data/title_emb_LFTM.txt'
	# emb_LFTM(datafile,vocabfile,outputfile)

	# datafile = '../data/train_text_emb.p'
	# vocabfile = '../data/news_train_text.p'
	# outputfile = '../data/text_emb_LFTM.txt'
	# emb_LFTM(datafile, vocabfile, outputfile)


