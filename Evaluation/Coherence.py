import _pickle as cPickle
import codecs
import math

class Document(object):
	def __init__(self):
		self.words = []
		self.length = 0

# 把整个文档及单词构成vocabulary（不允许重复）
class DataPreprocessing(object):
	def __init__(self):
		self.docs_count = 0  # 语料库中总文档数
		self.words_count = 0  # 词汇表中总词数
		# 保存每个文档d的信息，单词序列，以及length
		self.docs = []
		# 建立vocabulary表
		self.word2id = {}

class TopicModelEval(object):
	def __init__(self, trainloadpath):
		x_data = cPickle.load(open(trainloadpath, "rb"))
		train=x_data[0]
		train_lab = []
		self.wordtoix=x_data[2]
		self.ixtoword =  x_data[3]
		del x_data
		self.dpre=DataPreprocessing()
		self.dpre.docs_count=len(train)
		self.dpre.words_count=len(self.wordtoix)
		for i in range(len(train)):
			doc=Document()
			doc.words=train[i]
			doc.length=len(train[i])
			self.dpre.docs.append(doc)
		self.dpre.word2id=self.wordtoix

		self.wordcount={}
		self.cowordcount={}

	def loadtopNwords(self,topNloadpath):
		x_topN = codecs.open(topNloadpath, 'r')
		topN_indx = []
		for line in x_topN:
			if len(line.strip())==0:
				# print('line length=0')
				continue
			tokens = line.split()[1:]
			words = [w_.split('(')[0] for w_ in tokens]
			idx=[]
			for w in words:
				if w not in self.wordtoix:
					print(line)
					print(w+'not in aabcjdsvj')
				# print(w)
				idx.append(self.wordtoix[w])
			# idx = [self.wordtoix[w] for w in words]
			topN_indx.append(idx)
		x_topN.close()
		return topN_indx

	def getTopicCoherence(self,topN_indx):
		K=len(topN_indx)
		top_words_num = len(topN_indx[0])
		coherence = 0
		for x in range(K):
			twords = topN_indx[x]
			for wi in range(len(twords)):
				for wj in range(wi + 1, len(twords)):
					countij = 0
					counti = 0
					countj = 0
					for di in range(self.dpre.docs_count):
						if twords[wi] in self.dpre.docs[di].words:
							counti += 1
							if twords[wj] in self.dpre.docs[di].words:
								countij += 1
								countj += 1
						else:
							if twords[wj] in self.dpre.docs[di].words:
								countj += 1
					coherence += math.log((countij + 1) / (countj * counti) * self.dpre.docs_count)
		return coherence / K

	def getTopicCoherence1(self,topN_indx):
		K=len(topN_indx)
		N=20
		top_words_num = len(topN_indx[0])
		coherence = 0
		for x in range(K):
			twords = topN_indx[x]
			for wi in range(N):
				# print(x,wi,len(twords))
				if twords[wi] not in self.wordcount:
					self.wordcount[twords[wi]]=0
					for di in range(self.dpre.docs_count):
						if twords[wi] in self.dpre.docs[di].words:
							self.wordcount[twords[wi]]+=1
				counti=self.wordcount[twords[wi]]
				for wj in range(wi + 1, N):
					if twords[wj] not in self.wordcount:
						self.wordcount[twords[wj]]=0
						for di in range(self.dpre.docs_count):
							if twords[wj] in self.dpre.docs[di].words:
								self.wordcount[twords[wj]] += 1
					countj=self.wordcount[twords[wj]]

					if (twords[wi],twords[wj]) not in self.cowordcount:
						self.cowordcount[(twords[wi],twords[wj])]=0
						for di in range(self.dpre.docs_count):
							if twords[wj] in self.dpre.docs[di].words and twords[wj] in self.dpre.docs[di].words:
								self.cowordcount[(twords[wi], twords[wj])] += 1
						self.cowordcount[(twords[wj], twords[wi])]=self.cowordcount[(twords[wi],twords[wj])]
					countij=self.cowordcount[(twords[wi],twords[wj])]
					coherence += math.log((countij + 1) / (countj * counti) * self.dpre.docs_count)
		return coherence / K

if __name__=='__main__':
	trainloadpath='../data/classifydata/classifydata_index.p'
	topicE = TopicModelEval(trainloadpath)

	ldaprefix='../RE/LDA/k_'
	ldasuffix='/testLDA.topWords'

	dmmprefix='../RE/DMM/k_'
	dmmsuffix = '/testDMM.topWords'

	lfldaprefix='../RE/LFLDA/k_'
	lfldasuffix='/testLFLDA.topWords'

	lfdmmprefix='../RE/LFDMM/k_'
	lfdmmsuffix='/testLFDMM.topWords'

	ldatopN_path_list=[]
	dmmtopN_path_list=[]
	lfldatopN_path_list=[]
	lfdmmtopN_path_list=[]

	for k in [10,20,30,40]:
		ldatopN_path_list.append(ldaprefix+str(k)+ldasuffix)
		dmmtopN_path_list.append(dmmprefix+str(k)+dmmsuffix)
		lfldatopN_path_list.append(lfldaprefix + str(k) + lfldasuffix)
		lfdmmtopN_path_list.append(lfdmmprefix + str(k) + lfdmmsuffix)

	coherence_list=[]
	for path in ldatopN_path_list:
		topN_indx=topicE.loadtopNwords(path)
		coherence_list.append(topicE.getTopicCoherence1(topN_indx))
	# print('lda coherence: k=10,20,30,40; topN=20')
	print(coherence_list)

	coherence_list = []
	for path in dmmtopN_path_list:
		topN_indx = topicE.loadtopNwords(path)
		coherence_list.append(topicE.getTopicCoherence1(topN_indx))
	# print('dmm coherence: k=10,20,30,40; topN=20')
	print(coherence_list)

	coherence_list = []
	for path in lfldatopN_path_list:
		topN_indx = topicE.loadtopNwords(path)
		coherence_list.append(topicE.getTopicCoherence1(topN_indx))
	# print('lflda coherence: k=10,20,30,40; topN=20')
	print(coherence_list)

	coherence_list = []
	for path in lfdmmtopN_path_list:
		topN_indx = topicE.loadtopNwords(path)
		coherence_list.append(topicE.getTopicCoherence1(topN_indx))
	# print('lfdmm coherence: k=10,20,30,40; topN=20')
	print(coherence_list)








