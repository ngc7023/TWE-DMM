"""
Input: tweet_filtered.txt, tweet_filteredlabel.txt
Output: 不同比例的.p文件，划分数据集，test作为无标签集
"""
from gensim.models.word2vec import Word2Vec
import _pickle as cPickle
import pandas as pd
import numpy as np
import random
dataset = 'Tweet'
if(dataset=='Tweet'):
    filename = './inputdata/tweet_filtered.txt'
    labelfilename= './inputdata/tweet_filteredlabel.txt'
    encoding = 'utf-8'
    outputfilename = './outputdata/tweet_filtered'
    class_name = ['apple','google','microsoft','twitter']

elif(dataset=='N20short'):
	filename = './inputdata/N20short.txt'
	labelfilename= './inputdata/N20short_label.txt'
	encoding = 'utf-8'
	outputfilename = './outputdata/N20short'
	class_name = ['rec.autos', 'talk.politics.misc', 'sci.electronics', 'comp.sys.ibm.pc.hardware', 'talk.politics.guns',
               'sci.med', 'rec.motorcycles', 'soc.religion.christian', 'comp.sys.mac.hardware', 'comp.graphics',
               'sci.space', 'alt.atheism', 'rec.sport.baseball', 'comp.windows.x', 'talk.religion.misc',
               'comp.os.ms-windows.misc', 'misc.forsale', 'talk.politics.mideast', 'sci.crypt', 'rec.sport.hockey']

def precess(filename,encoding,outputfilename): # copy from TWE/preprocess_ch.py
	doccontent_list=[]
	doctext_list=[]
	f=open(filename,'r')
	ix=0
	max_len=0
	for line in f:
		doctext_list.append(line)
		token_list=line.split()
		if max_len<len(token_list):
			max_len=len(token_list)
		if len(token_list)!=0:
			doccontent_list.append(token_list)
			ix+=1
	f.close()
	print('docnumber:',ix)
	print('max_len:',max_len)
	vocab={}
	for doc in doccontent_list:
		for w in doc:
			if w in vocab:
				vocab[w]+=1
			else:
				vocab[w]=1

	# name_list = [k.lower().split('.') for k in class_name]
	# id_list = [ [ wordtoix[i] for i in l] for l in name_list]
	# value_list = [ [ opt.W_emb[i] for i in l]    for l in id_list]
	# value_mean = [ np.mean(l)  for l in id_list]

	# vocab1={}                # 过滤次数少的单词
	# for w in vocab:
	# 	if vocab[w]>2: # todo:参数确定【Tweet语料之前已经过滤了频率少于3的词】
	# 		vocab1[w]=vocab[w]
	# modify by pangjy
	# print('len_vocab1:', len(vocab))
	print('len_vocab0:', len(vocab))
	wordtoix = {}
	ixtoword = {}
	wordtoix['UNK'] = 0
	wordtoix['END'] = 1
	ixtoword[0] = 'UNK'
	ixtoword[1] = 'END'

	ix = 2 # 从2开始存
	for w in vocab:
		wordtoix[w] = ix
		ixtoword[ix] = w
		ix += 1

	name_list = [k.lower().split('.') for k in class_name]
	keys = wordtoix.keys()
	class_word_list = []
	for l in name_list:
		for i in l:
			if i in keys:
				continue
			else:
				class_word_list.append(i)
	cix = ix
	class_word_set = list(set(class_word_list))
	for i in class_word_set:
		wordtoix[i] = cix
		ixtoword[cix] = i
		cix += 1
	print(class_word_set)
	print("add class word into vocab, total number:", len(class_word_set))

	docs_ix_list = []
	count_noword = 0
	for doc in doccontent_list:
		docix_list = []
		for w in doc:
			if w in wordtoix:
				docix_list.append(wordtoix[w])
			else:
				print(w)
				count_noword+=1
				docix_list.append(wordtoix['UNK'])
		docs_ix_list.append(docix_list)
	return docs_ix_list, wordtoix, ixtoword

def process_label(labelfile):
	df = pd.read_csv(labelfile, names=['label'])
	label_list = df['label'].values.tolist()
	# number_of_labels = len(label_list)

	tmp_list = list(set(label_list))
	tmp_list.sort(key=label_list.index)
	label_dic = dict(zip(tmp_list, list(range(0, len(tmp_list)))))

	label_key_list = []
	for item in label_list:
		label_key_list.append(label_dic[item])

	return label_key_list,label_dic

def make_train_data(docs_ix_list,label_list,label_dic,prop_withlabel,train_prop,val_prop):
	# for i in label_list:
	# 	print(i)
	# 划分数据集

	number_list = []
	for i in range(len(label_dic)):
		key = list(label_dic.keys())[list(label_dic.values()).index(i)]
		number_list.append(label_list.count(label_dic[key]))

	train = []
	val = []
	test = []
	train_lab = []
	val_lab = []
	test_lab = []

	idx = 0

	# print(number_list)

	for num in number_list:
		current_docs_list = docs_ix_list[idx:idx+num]
		current_label_list = label_list[idx:idx+num]
		# print(idx,idx+num)

		# 把label转为LEAM用的one-hot格式 e.g. [0 0 1 0]
		for i in range(len(current_label_list)):
			one_digit_label = current_label_list[i]
			# print(one_digit_label)
			tmp = [0] * len(label_dic)
			tmp[one_digit_label] = 1
			current_label_list[i] = np.array(tmp)
			# current_label_list_tmp[i] = np.array(tmp)
		# 取train
		tmp_train = random.sample(current_docs_list, round(num*train_prop))
		train +=  tmp_train

		for item in tmp_train:
			idx_lab = current_docs_list.index(item)
			lab = current_label_list[idx_lab]
			train_lab.append(lab)
			# current_docs_list.remove(item)
			# current_label_list_tmp.pop(idx_lab)
		for item in tmp_train:
			idx_lab = current_docs_list.index(item)
			current_docs_list.pop(idx_lab)
			current_label_list.pop(idx_lab)

		tmp_val = random.sample(current_docs_list, round(num*val_prop))
		val += tmp_val

		for item in tmp_val:
			idx_lab = current_docs_list.index(item)
			lab = current_label_list[idx_lab]
			val_lab.append(lab)

		for item in tmp_val:
			idx_lab = current_docs_list.index(item)
			current_docs_list.pop(idx_lab)
			current_label_list.pop(idx_lab)
		# print(len(val),len(val_lab))

		test += current_docs_list
		test_lab += current_label_list

		# print(len(val),len(val_lab))
		idx+=num

	return train,val,test,train_lab,val_lab,test_lab

if __name__ == '__main__':
	print("processing file")
	docs_ix_list, wordtoix, ixtoword = precess(filename, encoding, outputfilename)
	print("processing label file")
	label_list,label_dic = process_label(labelfilename)

	proportion_withoutlabel_list = [0.7, 0.6, 0.5, 0.4, 0.3]
	train_proportion = [0.] * len(proportion_withoutlabel_list)
	val_proportion = [0.] * len(proportion_withoutlabel_list)

	for i in range(len(proportion_withoutlabel_list)):
		train_proportion[i] = (1. - proportion_withoutlabel_list[i]) * 0.9
		val_proportion[i] = (1. - proportion_withoutlabel_list[i]) * 0.1

	for i in range(len(proportion_withoutlabel_list)):
		prop_withoutlabel = proportion_withoutlabel_list[i]
		train_prop = train_proportion[i]
		val_prop = val_proportion[i]
		train, val, test, train_lab, val_lab, test_lab = make_train_data(docs_ix_list, label_list, label_dic, prop_withoutlabel, train_prop, val_prop)
		# print("len of train,val,test:", len(train), len(train_lab), len(val), len(val_lab), len(test), len(test_lab))
		if (len(test) % 2 == 1):  # 保证没有label的数据集的数量是偶数，因为LEAM的最小batch_size是2
			val.append(test[0])
			val_lab.append(test_lab[0])
			test = test[1:]
			test_lab = test_lab[1:]
		print("len of train,val,test:", len(train), len(train_lab), len(val), len(val_lab), len(test), len(test_lab))
		cPickle.dump([train, val, test, train_lab, val_lab, test_lab, wordtoix, ixtoword], open(outputfilename + str(prop_withoutlabel) + '.p', "wb"))
