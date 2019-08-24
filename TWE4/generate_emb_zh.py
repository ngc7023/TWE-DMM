"""
 @Time    : 2019/8/19 22:30
 @Author  : Pangjy
 @File    : TWE4/generate_emb_zh.py
 @Software: PyCharm
 Copy from TWE/generate_emb_zh.py
"""

import gensim
import numpy as np
import codecs
import _pickle as cPickle
filename = '../data/GoogleNews-vectors-negative300.bin'
# filename = '../data/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(filename,binary=True)

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
	file = open("langdetect_tweet.wordVectors" + '.txt', 'w')
	for word in wordtoix.keys():
		idx = wordtoix.get(word)
		if(word!='UNK'):
			if word in glove_vocab:
				embedding_vectors[idx] = model.wv[word]
				emb_str = ' '.join(str(i) for i in embedding_vectors[idx])
				file.write(str(word)+" "+emb_str)
				file.write('\n')
				count += 1
			else:
				emb_str = ' '.join(str(i) for i in embedding_vectors[idx])
				file.write(str(word)+" "+emb_str)
				file.write('\n')
				mis_count += 1
	# file = open("langdetect_tweet.wordVectors" + '.txt', 'w')
	# 		file.write(str(row.name))
	# 		file.write('\n')
	# 	# print("writing")
	# file.close()

	print("num of vocab in word2vec: {}".format(count))
	print("num of vocab not in word2vec: {}".format(mis_count))
	# print("num of vocab in glove: {}".format(count))
	# print("num of vocab not in glove: {}".format(mis_count))
	file.close()
	exit()
	cPickle.dump([embedding_vectors],open(outputfile, 'wb'))
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
	outputfile = '../data/TACL-datasets/langdetect_tweet.p'
	generate_emb(datafile, outputfile)

	# datafile = '../data/TACL-datasets/TMNtitle.p'
	# outputfile = '../data/TACL-datasets/TMNtitle_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile = '../data/TACL-datasets/TMNfull.p'
	# outputfile = '../data/TACL-datasets/TMNfull_emb.p'
	# generate_emb(datafile, outputfile)
    #
	# datafile = '../data/TACL-datasets/N20small.p'
	# outputfile = '../data/TACL-datasets/N20small_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile = '../data/TACL-datasets/N20.p'
	# outputfile = '../data/TACL-datasets/N20_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile = '../data/TACL-datasets/langdetect_tweet.p'
	# outputfile = '../data/TACL-datasets/langdetect_tweet_emb.p'
	# generate_emb(datafile, outputfile)

	# datafile='../data/TACL-datasets/N20short.p'
	# outputfile='../data/TACL-datasets/N20short_emb.p'
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


