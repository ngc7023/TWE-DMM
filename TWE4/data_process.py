import numpy as np
def make_train_data_0(train,length):
    id_list=[0]*(length-1) # len(id_list)=29
    for i in range(len(train)):
        id_list+=train[i]
    fea=[]
    lab=[]
    i=0

    while i<len(id_list):
        if i+length>=len(id_list):
            fea.append(id_list[i:]+[0]*(length-len(id_list[i:])))
            lab.append(0)
        else:
            fea.append(id_list[i:i+length])
            lab.append(id_list[i+length])
        i+=1
    return fea,lab

def make_train_data_0722(train): # 2019-07-22
    label=[]
    sample_num=0
    for i in range(len(train)): # 每个文本
        len_text = len(train[i]) # 文本长度
        for j in range(1,len_text): # 每个单词
            label.append([i,j])
            sample_num+=1
    return label,sample_num


def make_train_data(train,dmm): # 2019-07-26
    len_train = len(train)
    label = [0 for i in range(0,dmm.sample_num)] # 初始化定长list
    ProbIdx_start = np.zeros(dmm.dpre.docs_count, dtype='int')
    sample_num = 0
    for i in range(len_train): # 每个文本
        len_text = len(train[i]) # 文本长度
        ProbIdx_start[i] = sample_num
        for j in range(1,len_text): # 每个单词
            label[sample_num] = [i,j]
            sample_num+=1
    return label, sample_num, ProbIdx_start
def make_train_data(train,opt): # 2019-08-19 PJY (for english corpus)
    label = [0] * opt.setSampleNumber
    len_train = len(train)
    sample_num = 0
    for i in range(len_train):  # 每个文本
        len_text = len(train[i])  # 文本长度
        for j in range(len_text):  # 每个单词
            label[sample_num] = [i,j]
            sample_num += 1
    return label, sample_num

def get_train_by_label(labix,maxlen,wordseq_row): # 2019-07-26 不拼接，补0
    if(labix < maxlen):
        word_sequence = [0]*(maxlen-labix) + wordseq_row[:labix+1]
    else:
        word_sequence = wordseq_row[labix-maxlen:labix+1]
    return word_sequence

def get_train_by_label1(label,maxlen,allwordseq): # 2019-08-05 拼接
    row = label[0]
    column = label[1]
    seq = allwordseq[row].words[:column+1]
    if(row==0):
        seq =[0]*(maxlen-len(seq)+1) + seq
        return seq
    else:
        i=1
        while(len(seq)<maxlen+1):
            seq = allwordseq[row-i].words+seq
            i+=1
        seq = seq[len(seq)-maxlen-1:]
        return seq
