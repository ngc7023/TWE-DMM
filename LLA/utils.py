import numpy as np

def read_data(train,train_lab):
	id_lsit=[]
	lab_list=[]
	for i in range(len(train)):
		id_lsit+=train[i]
		lab_list+=train_lab[i]
	return id_lsit,lab_list

def make_batches(train,train_lab,batches_size,num_step):
	id_list,lab_list=read_data(train,train_lab)
	total_words=len(id_list)

	data=[]
	label=[]
	i=0
	while i<len(id_list):
		tmpbatch=[]
		tmplabel=[]
		for j in range(batches_size):
			if i<len(id_list) and i+num_step>len(id_list):
				tmpbatch.append(id_list[i:]+[0]*(num_step-len(id_list[i:])))
				tmplabel.append(lab_list[i:]+[0]*(num_step-len(lab_list[i:])))
				i+=num_step
			elif i>=len(id_list):
				tmpbatch.append([0]*num_step)
				tmplabel.append([0]*num_step)
			else:
				tmpbatch.append(id_list[i:i+num_step])
				tmplabel.append(lab_list[i:i+num_step])
				i+=num_step
		data.append(tmpbatch)
		label.append(tmplabel)
	data=np.array(data)
	label=np.array(label)
	return data,label,total_words

def read_data_topic(train_lab):
	lab_list=[]
	for i in range(len(train_lab)):
		lab_list+=train_lab[i]
	return lab_list

def make_batches_topic(train,train_lab,batches_size,num_step):
	lab_list=read_data_topic(train_lab)
	total_words=len(lab_list)-1

	data=[]
	label=[]
	i=0
	while i<len(lab_list):
		tmpbatch=[]
		tmplabel=[]
		for j in range(batches_size):
			if i<len(lab_list) and i+num_step>len(lab_list):
				tmpbatch.append(lab_list[i:]+[0]*(num_step-len(lab_list[i:])))
				tmplabel.append(lab_list[i+1:]+[0]*(num_step-len(lab_list[i+1:])))
				i+=num_step
			elif i>=len(lab_list):
				tmpbatch.append([0]*num_step)
				tmplabel.append([0]*num_step)
			else:
				tmpbatch.append(lab_list[i:i+num_step])
				tmplabel.append(lab_list[i+1:i+1+num_step])
				i+=num_step
		data.append(tmpbatch)
		label.append(tmplabel)
	data=np.array(data)
	label=np.array(label)
	return data,label,total_words