import _pickle as cPickle
import os
import codecs
import numpy as np

def vis_att(datafile,attfile):
	x=cPickle.load(open(datafile,'rb'))
	train=x[0]
	label=x[1]
	wordtoix, ixtoword=x[2],x[3]
	del x
	for i in range(len(label)):
		if '1330' in label[i] and '体育' in label[i]:
			tiyu=i
			print(i,label[i])
		if '164' in label[i] and '校园' in label[i]:
			xiaoyuan=i
			print(i,label[i])

	x=cPickle.load(open(attfile,'rb'))
	attention=x[0]
	del x
	words=[ixtoword[idx] for idx in train[tiyu]]
	# print(attention[tiyu])
	tmp=[]
	for i in range(len(train[tiyu])):
		tmp.append(attention[tiyu][i][0])
	tmp=np.array(tmp)*100
	print('体育：')
	for i in range(len(tmp)):
		print(words[i],tmp[i])
	print('\n\n')
	words=[ixtoword[idx] for idx in train[xiaoyuan]]
	tmp = []
	for i in range(len(train[xiaoyuan])):
		tmp.append(attention[xiaoyuan][i][0])
	tmp = np.array(tmp) * 100
	print('校园：')
	for i in range(len(tmp)):
		print(words[i], tmp[i])

def readText():
	inputPath = "../data/classifydata/data/"
	lenlist=[]
	fatherLists = os.listdir(inputPath)  # 主目录
	for eachDir in fatherLists:  # 遍历主目录中各个文件夹
		if eachDir[0]=='.':
			continue
		eachPath = inputPath + eachDir + "/"  # 保存主目录中每个文件夹目录，便于遍历二级文件
		print(eachDir)
		childLists = os.listdir(eachPath)  # 获取每个文件夹中的各个文件
		for eachFile in childLists:  # 遍历每个文件夹中的子文件
			eachPathFile = eachPath + eachFile  # 获得每个文件路径
			f=codecs.open(eachPathFile,encoding = 'GBK')
			# print(eachPathFile)
			text_string=''
			for line in f:
				text_string+=line.strip()
			lenlist.append(len(text_string))
			print(eachFile,text_string)
		print('--------------------------------------------------------------------------------------------------------------')
		print('\n\n')

	print(min(lenlist),max(lenlist))

if __name__ == '__main__':
	readText()

	# datafile = '../data/classifydata/classifydata_index.p'
	# attfile='../RE/TWE_DMM/k_4/attention.p'
	# vis_att(datafile,attfile)