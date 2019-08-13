import _pickle as cPickle
import codecs
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
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

def gettopNwords(datafile,N):
	topic_words={}
	f=codecs.open(datafile,'r')
	ix=0
	for line in f:
		tokens=line.strip().split()
		twords=[(n,tokens[n]) for n in range(len(tokens))]
		twords.sort(key = lambda i: i[1], reverse = True)
		topNindex=[tw[0] for tw in twords[:N]]
		topic_words[ix]=topNindex
		ix+=1
		# print(topNindex)
	f.close()
	return topic_words

def loademb(embfile,topic_words):
	x=cPickle.load(open(embfile,'rb'))
	wordemb=x[0]
	topicemb=x[1]
	del x
	topicname=[]
	for k in range(len(topicemb)):
		topicname.append('Topic '+str(k+1))
	wordemb_=[]
	target_w_=[]
	for k in range(len(topic_words)): # len（topic_words）= topic_number
		for wi in topic_words[k]:
			# target_w[wi]=k+1
			wordemb_.append(wordemb[wi])
			target_w_.append(k)
	target_t=[k for k in range(len(topicemb))]
	topicwordemb=wordemb_+list(topicemb)
	X_tsne=TSNE(learning_rate = 100,metric='cosine',early_exaggeration=12).fit_transform(topicwordemb)
	X_word=X_tsne[:len(wordemb_)]
	X_topic=X_tsne[len(wordemb_):]

	fig, _ = plt.subplots()
	plt.subplot(111)
	plt.scatter(X_word[:, 0], X_word[:, 1], marker='.',s=20, c = target_w_)
	plt.scatter(X_topic[:, 0], X_topic[:, 1], marker = 'o',s=60, c = target_t,edgecolors = 'k')
	plt.title('t-SNE plot of topic word embedding')
	plt.xticks([])
	plt.yticks([])

	# plt.subplot(122)
	# y=list(range(len(topic_words)))
	# x=[1 for _ in range(len(topic_words))]
	# plt.scatter(x,y,c=target_t,marker = 'o',s=20,linewidths = 5,)
	# fig.legend(tuple(ax_list),tuple(topicname),loc = 'left center', bbox_to_anchor = (0.5, 0.89), ncol = 1, columnspacing = 0.1)
	plt.show()

def getsimwords(embfile,N): # 计算余弦相似度
	x = cPickle.load(open(embfile, 'rb'))
	wordemb = x[0]
	topicemb = x[1]
	del x
	topicsimword = {}
	it = 0
	for temb in topicemb: # 对于每个topic
		sim_list = []
		ix = 0
		for wemb in wordemb: # 对于每个word
			cosinesim = cos_sim(temb, wemb) # 计算余弦相似度
			sim_list.append((ix, cosinesim))
			ix += 1
		sim_list.sort(key = lambda i: i[1], reverse = True) # 排序
		# topicsimword[it] = [w[0] for w in sim_list[:N]] # 每个topic的
		# it +=1
		threshold = 0
		thresholdix = 0
		for j in range(len(sim_list)):
			if(sim_list[j][1]>0.7):
				threshold=sim_list[j][1]
				thresholdix = sim_list[j][0]

	for temb in topicemb:  # 对于每个topic
		sim_list2 = []
		ix = 0
		for wemb in wordemb: # 对于每个word
			cosinesim = cos_sim(temb, wemb) # 计算余弦相似度
			ixinput = ix
			if(cosinesim<0.7):
				cosinesim = threshold
				ixinput = thresholdix
			sim_list2.append((ixinput, cosinesim))
			ix += 1
		sim_list2.sort(key = lambda i: i[1], reverse = True) # 排序
		topicsimword[it] = [w[0] for w in sim_list2[:N]] # 每个topic的
		print(sim_list2[:N])
		it += 1
	return topicsimword

if __name__=='__main__':
	phifile='../RE/TWE_DMM/k_10/phifile.txt'
	# topic_words=gettopNwords(phifile,500)
	embfile='../RE/TWE_DMM/k_10/topic-wordemb.p'
	topic_words=getsimwords(embfile,100) # 为所有topic找他们最相近的word
	loademb(embfile,topic_words)

	# cPickle.dump([word_emb,topic_emb],open('./re/topic-wordemb.p','wb'))
	# W = tf.get_variable('W', initializer = opt.W_emb, trainable = True)  # initializer: 指定初始化方式
	#