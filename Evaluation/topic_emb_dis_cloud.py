import _pickle as cPickle
import numpy as np
import codecs
import matplotlib.pyplot as plt

def cos_sim(vector_a, vector_b):
	"""
	计算两个向量之间的余弦相似度
	:param vector_a: 向量 a
	:param vector_b: 向量 b
	:return: sim
	"""
	vector_a = np.mat(vector_a)
	vector_b = np.mat(vector_b)
	num = float(vector_a * vector_b.T)
	denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
	cos = num / denom
	sim = 0.5 + 0.5 * cos
	return sim

def loademb(embfile):
	x=cPickle.load(open(embfile,'rb'))
	wordemb=x[0]
	topicemb=x[1]
	del x
	return wordemb,topicemb

def loadcorpus(corpusfile):
	x = cPickle.load(open(corpusfile, 'rb'))
	wordtoix = x[2]
	ixtoword = x[3]
	del x
	return ixtoword

def gettopsim(wordemb,topicemb,ixtoword,N):
	topicsimword = {}
	it = 0
	for temb in topicemb:
		sim_list = []
		ix = 0
		for wemb in wordemb:
			sim_list.append((ix, cos_sim(temb, wemb)))
			ix += 1
		sim_list.sort(key = lambda i: i[1], reverse = True)
		topicsimword[it] = [(ixtoword[w[0]],w[1]) for w in sim_list[:N]]
		it += 1
	return topicsimword

def gettopprob(phifile,ixtoword,N):
	f = codecs.open(phifile, 'r')
	topicprobword = {}
	it=0
	for line in f:
		tokens = line.strip().split()
		prob = [float(p) for p in tokens]
		twords = [(ixtoword[n], prob[n]) for n in range(len(prob))]
		twords.sort(key = lambda i: i[1], reverse = True)
		topicprobword[it]=twords[:N]
		it+=1
	f.close()
	return topicprobword

if __name__ == '__main__':
	embfile = '../TWE/re/k_30/topic-wordemb.p'
	phifile = '../TWE/re/k_30/phifile.txt'
	corpusfile = '../data/classifydata/classifydata_index.p'
	wordemb, topicemb = loademb(embfile)
	ixtoword=loadcorpus(corpusfile)
	N=100
	topicsimword=gettopsim(wordemb,topicemb,ixtoword,N)
	topicprobword=gettopprob(phifile,ixtoword,N)
	for i in range(len(topicsimword)):
		print('topic:',i)
		tmp=np.array([t[1] for t in topicprobword[i]])
		tmp=tmp/sum(tmp)
		tmp=[round(x,5) for x in tmp]
		print([(topicprobword[i][j][0],tmp[j]) for j in range(len(topicprobword[i]))])

		tmp = np.array([t[1] for t in topicsimword[i]])
		tmp = tmp/ sum(tmp)
		tmp = [round(x, 5) for x in tmp]
		print([(topicsimword[i][j][0], tmp[j]) for j in range(len(topicsimword[i]))])
		print('==============================================================================')