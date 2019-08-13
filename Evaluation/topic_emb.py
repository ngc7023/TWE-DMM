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

def loadphi(phifile):
	f=codecs.open(phifile,'r')
	phi=[]
	for line in f:
		tokens=line.strip().split()
		prob=[float(p) for p in tokens]
		twords = [(n, prob[n]) for n in range(len(prob))]
		twords.sort(key = lambda i: i[1], reverse = True)
		phi.append(twords)
	return phi

def topic_emb_value(wordemb,topicemb,phi):
	re=[]
	for i in range(len(phi)):  # 打印所有数据
		print('topic:',i)
		te=topicemb[i] # topic-embedding
		value=[]
		count=0
		tmp=np.zeros(len(topicemb[0]))
		for w in phi[i]:
			tmp+=wordemb[w[0]]
			count+=1
			if count%20==0:
				value.append(cos_sim(te,tmp))
			if count>=100:
				break
		re.append(value)
		print(value)
	return re

def plot_topic_value(re):
	re=np.array(re)
	re=re.T
	label=['Top-20','Top-40','Top-60','Top-80','Top-100']
	linestyle=['-','-.',':','--',':.']
	fig, ax = plt.subplots()
	plt.subplot(111)
	x=list(range(1,len(re[0])+1)) # x就是一个下标

	for i in range(len(re)):
		y=sorted(re[i],reverse =True)
		plt.plot(x,y,linestyle[i],label=label[i])
	plt.xlabel('Sorted topic index')
	plt.ylabel('Cosine similarity')
	plt.ylim(0,1)
	plt.xlim(0,41)
	fig.legend(loc = 'upper center', ncol = 3, bbox_to_anchor = (0.5, 0.89), columnspacing = 0.1)
	plt.show()

if __name__=='__main__':
	embfile='../RE/TWE_DMM/k_40/topic-wordemb.p'
	phifile='../RE/TWE_DMM/k_40/phifile.txt'
	wordemb,topicemb=loademb(embfile)
	phi=loadphi(phifile)
	re=topic_emb_value(wordemb,topicemb,phi)
	plot_topic_value(re)




