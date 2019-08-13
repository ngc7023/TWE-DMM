import _pickle as cPickle
import numpy as np

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

def loademb(embfile,corpusfile):
	N=20
	x=cPickle.load(open(embfile,'rb'))
	wordemb=x[0]
	topicemb=x[1]
	del x
	x=cPickle.load(open(corpusfile,'rb'))
	wordtoix=x[2]
	ixtoword=x[3]
	del x
	topicsimword={}
	it=0
	for temb in topicemb:
		sim_list=[]
		ix=0
		for wemb in wordemb:
			sim_list.append((ix, cos_sim(temb,wemb)))
			ix+=1
		sim_list.sort(key = lambda i: i[1], reverse = True)
		topicsimword[it]=[w[0] for w in sim_list[:N]]
		it+=1
	return topicsimword



if __name__=='__main__':
	embfile='../TWE/re/k_40/topic-wordemb.p'
	corpusfile='../data/classifydata/classifydata_index.p'
	loademb(embfile,corpusfile)