# -*- coding: utf-8 -*-
import codecs
import re
import itertools

import _pickle as cPickle
import jieba

# 加载自定义词典
# jieba.load_userdict(file_name)
# jieba.cut(sq) # 分词

# 从文件导入停用词表
stpwrdpath = "../data/stop_word2.txt"
stpwrd_dic = codecs.open(stpwrdpath, encoding = 'utf-8')
stpwrd_content = stpwrd_dic.read()
# 将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

# 加载搜狗新闻数据类别标签
target_namefile='../data/SogouTCE.txt'
target_name={}
target_f=codecs.open(target_namefile,encoding = 'GB18030')
name=[]
for line in target_f:
	tokens=line.split()
	target_name[tokens[0]]=tokens[1]
	if tokens[1] not in name:
		name.append(tokens[1])
target_f.close()
print(len(name),name)

def precess(filename,encoding,outputfilename):
	doctitle_list=[]
	doccontent_list=[]
	doctarget_list=[]
	f=codecs.open(filename,encoding = encoding)
	for line in f:
		if re.match('<url>',line):
			r_=re.search('^<url>(.*)</url>$',line)
			url=r_.group(1)
			flg=False
			for t in target_name:
				if t in url:
					doctarget_list.append(target_name[t])
					flg=True
					break
			if flg==False:
				doctarget_list.append('UNK')
				print('类别未找到!',url)
		if re.match('<contenttitle>',line):
			doctitle=re.search('^<contenttitle>(.*)</contenttitle>$',line)
			token_list=[w for w in jieba.cut(doctitle.group(1)) if w not in stpwrdlst]
			doctitle_list.append(token_list)
		if re.match('<content>',line):
			doccontent=re.search('^<content>(.*)</content>$',line)
			token_list=[w for w in jieba.cut(doccontent.group(1)) if w not in stpwrdlst]
			doccontent_list.append(token_list)
	f.close()

	cPickle.dump([doctitle_list,doctarget_list,doccontent_list], open(outputfilename, "wb"))

def precess1(filename,encoding,outputfilename):
	docno_list=[]
	doccontent_list=[]
	doctext_list=[]
	f=codecs.open(filename,'r')
	ix=0
	max_len=0
	for line in f:
		doctext_list.append(line)
		token_list=line.split()
		if max_len<len(token_list):
			max_len=len(token_list)
		if len(token_list)!=0:
			doccontent_list.append(token_list)
			docno_list.append(ix)
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
	vocab1={}
	for w in vocab:
		if vocab[w]>20:
			vocab1[w]=vocab[w]
	print('len_vocab:',len(vocab1))

	wordtoix = {}
	ixtoword = {}
	wordtoix['UNK'] = 0
	ixtoword[0] = 'UNK'
	ix = 1
	for w in vocab1:
		wordtoix[w] = ix
		ixtoword[ix] = w
		ix += 1

	doccententindex = []
	for doc in doccontent_list:
		docix_list = []
		for w in doc:
			if w in wordtoix:
				docix_list.append(wordtoix[w])
			else:
				docix_list.append(wordtoix['UNK'])
		doccententindex.append(docix_list)
	# print(len(doccontent_list),len(docno_list))

	cPickle.dump([doccententindex,docno_list,wordtoix, ixtoword,doctext_list], open(outputfilename, "wb"))

def precess2():
	maxlen=0
	f=codecs.open('../data/corpus_text.txt','r')
	doccontent_list = []
	ix=0
	for line in f:
		token_list = line.strip().split()
		if len(token_list)!=0:
			doccontent_list.append(token_list)
			# if len(token_list)>100:
			# 	print(ix,len(itertools),token_list)
			if maxlen<len(token_list):
				maxlen=len(token_list)
			ix+=1
	print('docnumber:',ix)
	f.close()
	print('max_len',maxlen)
	vocab = {}
	for doc in doccontent_list:
		for w in doc:
			if w in vocab:
				vocab[w] += 1
			else:
				vocab[w] = 1
	vocab1 = {}
	for w in vocab:
		if vocab[w] > 20:
			vocab1[w] = vocab[w]
	print('len_vocab',len(vocab1))

	outf = codecs.open('../data/corpus_text_less20.txt', 'a')
	for doc in doccontent_list:
		doc_string=''
		for w in doc:
			if w in vocab1:
				doc_string+=w+' '
			else:
				doc_string+='UNK'+' '
		doc_string+='\n'
		outf.write(doc_string)
	outf.close()

def test():
	doccount=0
	lenmax=0
	f = codecs.open('../data/新闻标题数据集/train_text.txt', encoding = 'utf-8')
	outf = codecs.open('../data/corpus_text.txt', 'w')
	for line in f:
		token_list=[w for w in jieba.cut(line.strip()) if w not in stpwrdlst]
		if lenmax<len(token_list):
			lenmax=len(token_list)
		if len(token_list):
			outf.write(' '.join(token_list)+'\n')
			doccount+=1
	f.close()
	outf.close()
	print(doccount,lenmax)




if __name__=='__main__':
    filename='../data/news_tensite_xml.dat'
    encoding='GB18030'
    outputfilename='../data/news_sogou.p'
    precess(filename,encoding,outputfilename)

    # filename='../data/corpus_text.txt'
    # encoding='utf-8'
    # outputfilename='../data/news_train_text.p'
    # precess1(filename,encoding,outputfilename)

    # filename = '../data/corpus_title.txt'
    # encoding = 'utf-8'
    # outputfilename = '../data/news_train_title.p'
    # precess1(filename, encoding, outputfilename)

    # precess2()
    # test()
