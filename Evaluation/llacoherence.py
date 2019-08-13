from Evaluation.Coherence import *

def getTopNindex(topNloadpath):
	x_topN = codecs.open(topNloadpath, 'r')
	topN_indx = []
	for line in x_topN:
		if len(line.strip()) == 0:
			# print('line length=0')
			continue
		tokens = line.strip().split(' ')[1:]
		idx= [int(w_.split('(')[0]) for w_ in tokens]
		topN_indx.append(idx)
	x_topN.close()
	# print(topN_indx)
	return topN_indx


if __name__ == '__main__':
	trainloadpath = '../data/classifydata/classifydata_index.p'
	topicE = TopicModelEval(trainloadpath)
	llaprefix = '../RE/LLA_topic/k_'
	# llaprefix='../RE/TWE_DMM/k_'
	llasuffix = '/topNfile.txt'

	llatopN_path_list = []
	for k in [10,20,30,40]:
		llatopN_path_list.append(llaprefix+str(k)+llasuffix)

	coherence_list = []
	for path in llatopN_path_list:
		topN_indx = getTopNindex(path)
		coherence_list.append(topicE.getTopicCoherence1(topN_indx))
	# print('lda coherence: k=10,20,30,40; topN=20')
	print(coherence_list)