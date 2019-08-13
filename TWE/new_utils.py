import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import pickle as cPickle
import sys
import os

def prepare_data_for_emb(seqs_x, opt):
	maxlen = opt.maxlen
	lengths_x = [len(s) for s in seqs_x]
	if maxlen != None:
		new_seqs_x = []
		new_lengths_x = []
		for l_x, s_x in zip(lengths_x, seqs_x):
			if l_x < maxlen:
				new_seqs_x.append(s_x)
				new_lengths_x.append(l_x)
			else:
				new_seqs_x.append(s_x[:maxlen])
				new_lengths_x.append(maxlen)
		lengths_x = new_lengths_x
		seqs_x = new_seqs_x

		if len(lengths_x) < 1:
			return None, None

	n_samples = len(seqs_x)
	maxlen_x = np.max(lengths_x)
	x = np.zeros((n_samples, maxlen)).astype('int32')
	x_mask = np.zeros((n_samples, maxlen)).astype('float32')
	for idx, s_x in enumerate(seqs_x):  # enumerate() ：同时列出数据下标和数据。
		x[idx, :lengths_x[idx]] = s_x
		x_mask[idx, :lengths_x[idx]] = 1.  # change to remove the real END token
	return x, x_mask

def restore_from_save(t_vars, sess, opt):
	save_keys = tensors_key_in_file(opt.save_path)
	# print(save_keys.keys())
	ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
	cc = {var.name: var for var in t_vars}
	ss_right_shape = set(
		[s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])  # only restore variables with correct shape

	if opt.reuse_discrimination:
		ss2 = set([var.name[2:] for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
		cc2 = {var.name[2:][:-2]: var for var in t_vars if var.name[2:] in ss2 if
			   var.get_shape() == save_keys[var.name[2:][:-2]]}
		for s_iter in ss_right_shape:
			cc2[s_iter[:-2]] = cc[s_iter]

		loader = tf.train.Saver(var_list = cc2)
		loader.restore(sess, opt.save_path)
		print("Loaded variables for discriminator:" + str(cc2.keys()))

	else:
		loader = tf.train.Saver(var_list = [var for var in t_vars if var.name in ss_right_shape])
		loader.restore(sess, opt.save_path)
		print("Loading variables from '%s'." % opt.save_path)
		print("Loaded variables:" + str(ss_right_shape))

	return loader

def tensors_key_in_file(file_name):
	"""Return tensors key in a checkpoint file.
	Args:
	file_name: Name of the checkpoint file.
	"""
	try:
		reader = pywrap_tensorflow.NewCheckpointReader(file_name)
		return reader.get_variable_to_shape_map()
	except Exception as e:  # pylint: disable=broad-except
		print(str(e))
		return None

def get_minibatches_idx(n, minibatch_size, shuffle = False):
	idx_list = np.arange(n, dtype = "int32")

	if shuffle:
		np.random.shuffle(idx_list)

	minibatches = []
	minibatch_start = 0
	for i in range(n // minibatch_size):  # // ：整除，取整数部分
		minibatches.append(idx_list[minibatch_start:
									minibatch_start + minibatch_size])
		minibatch_start += minibatch_size
	return zip(range(len(minibatches)), minibatches)

def load_class_embedding(wordtoidx, opt):
	print("load class embedding")
	name_list = [k.lower().split(' ') for k in opt.class_name]
	id_list = [[wordtoidx[i] for i in l] for l in name_list]
	value_list = [[opt.W_emb[i] for i in l] for l in id_list]
	value_mean = [np.mean(l, 0) for l in value_list]
	return np.asarray(value_mean)

#
# def make_train_data(train,length):
# 	id_list=[0]*(length-1) # len(id_list)=29
# 	# print(len(train))
# 	for i in range(len(train)):
# 		id_list+=train[i]
# 	fea=[]
# 	lab=[]
# 	i=0
# 	# print(len(id_list))
# 	while i<len(id_list):
# 		if i+length>=len(id_list):
# 			fea.append(id_list[i:]+[0]*(length-len(id_list[i:])))
# 			lab.append(0)
# 		else:
# 			fea.append(id_list[i:i+length])
# 			lab.append(id_list[i+length])
# 		i+=1
# 	return fea,lab

def make_train_data(train):
	label=[]
	sample_num=0
	for i in range(len(train)): # 每个文本
		len_text = len(train[i]) # 文本长度
		for j in range(1,len_text): # 每个单词
			label.append([i,j])
			sample_num+=1
	return label,sample_num

def get_train_by_label(label,maxlen,wordseq): # modify by pjy
	word_sequence = [0]*maxlen
	tmp = [0]*maxlen
	labix = label[1].astype(np.int)

	if(labix<maxlen):
		tmp=wordseq[:labix]
		for j in range(labix):
			word_sequence[maxlen-labix+j]=tmp[j]
	else:
		word_sequence = wordseq[labix-30:labix]
	return word_sequence

def get_train_by_label1(label,maxlen,allwordseq): # 2019-08-05 modifybypjy
    row = label[0]
    column = label[1]
    seq = allwordseq[row].words[:column+1]
    i=1
    while(len(seq)<maxlen+1):
        seq = allwordseq[row-i].words+seq
        i+=1
    seq = seq[len(seq)-maxlen-1:]
    return seq
