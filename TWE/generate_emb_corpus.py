import _pickle as cPickle
import random

def generate(datafile,outputfile,topicNumber):
	loadpath = datafile
	x = cPickle.load(open(loadpath, "rb"))
	train= x[0]
	train_lab =x[1]
	wordtoix, ixtoword = x[2], x[3]

	samples=[]
	w_labels=[]
	topic_lab=[]
	for ix,doc in enumerate(train):
		for i in range(len(doc)):
			samples.append(doc[0:i]+doc[i+1:])
			w_labels.append(doc[i])
			topic_lab.append(train_lab[ix])

	cPickle.dump([samples, topic_lab,w_labels, wordtoix, ixtoword], open(outputfile, "wb"))

if __name__=='__main__':
	# datafile='../data/news_train_text.p'
	# outputfile='../data/train_text_corpus.p'
	# generate(datafile,outputfile)

	datafile = '../data/news_train_title.p'
	outputfile = '../data/train_title_corpus.p'
	generate(datafile, outputfile)


