import pandas as pd
import numpy as np
class MustSets(object):
    def __init__(self):
        self.mustsets = []
        self.len_mustsets = []
        self.num_doc_mustsets = []
        self.wordstrToMustsetListMap = dict()
        self.wordidToMustsetListMap = dict()
        self.wordidToMustsetListMapForNoLabel = dict()
        self.threshold = 0.1 # 过滤attention_score
        self.normalized_size = None

        # MS * len(mset)
        self.realtedword = None
        self.frequency = None
        self.MS = None
        # for filter word in mustset, ngram
        self.r = 2

    def InitMustSets(self,vocab_dict,text,labelfile):
        # 只能处理label已经是顺序的文件，corpus和label是对应的
        df = pd.read_csv(labelfile, names=['label'])
        label_list = df['label'].values.tolist()
        tmp_list = list(set(label_list))
        tmp_list.sort(key=label_list.index)
        label_dic = dict(zip(tmp_list, list(range(0, len(tmp_list)))))

        self.mustsets = [[] for i in range(len(label_dic))]
        tmp_mustsets = [[] for i in range(len(label_dic))]

        idx_label = 0
        for line in text:
            for key in label_dic.keys():
                if(label_list[idx_label]==key):
                    tmp_mustsets[label_dic[key]] += line
                    idx_label+=1
                    break

        for i in range(len(label_dic)):
            self.mustsets[i] = list(set(tmp_mustsets[i]))
        self.length = len(self.mustsets)
        self.addWordIntoMap(vocab_dict)

    def addWordIntoMap(self,vocab_dict):
        for wordstr in list(vocab_dict.keys())[1:]:
            self.wordstrToMustsetListMap[wordstr] = []
            for i in range(len(self.mustsets)):
                if(wordstr in self.mustsets[i]):
                    self.wordstrToMustsetListMap.get(wordstr).append(i)

    def addWordIntoMapUseLEAM(self,ixtoword):
        for word_id in list(ixtoword.keys()):
            self.wordidToMustsetListMap[word_id] = []
            for i in range(len(self.mustsets)):
                if(word_id in self.mustsets[i]):
                    self.wordidToMustsetListMap.get(word_id).append(i)

    def addWordIntoMapForNoLabel(self,text_line,j):
        for word_id in text_line:
            if j not in self.wordidToMustsetListMapForNoLabel.get(word_id):
                self.wordidToMustsetListMapForNoLabel.get(word_id).append(j)

    def getMustSetListGivenWordid(self, wordid):
        return self.wordidToMustsetListMap.get(wordid)

    def getMustSetListGivenNoLabelWordid(self, wordid):
        return self.wordidToMustsetListMapForNoLabel.get(wordid)

    def InitMustsetUseLEAM(self, train, train_lab, test, test_lab, wordtoix, ixtoword):
        # label_list = train_lab
        self.MS = len(train_lab[0])

        self.mustsets = [[] for i in range(self.MS)]
        tmp_mustsets = [[] for i in range(self.MS)]

        for i in range(len(train)):
            key = list(train_lab[i]).index(1) # 确定label
            tmp_mustsets[key] += train[i]

        for i in range(self.MS):
            self.mustsets[i] = list(set(tmp_mustsets[i]))

        self.addWordIntoMapUseLEAM(ixtoword)

    def ExtendMustSetsWithPredictLabelText(self, test, test_lab, ixtoword):
        tmp_mustsets = self.mustsets

        for word_id in list(ixtoword.keys()):
            self.wordidToMustsetListMapForNoLabel[word_id] = []

        for i in range(len(test)):
            for j in range(self.MS):
                if(test_lab[i][j]!=0):
                    self.addWordIntoMapForNoLabel(test[i],j)
                    tmp_mustsets[j] += test[i]

        for i in range(self.MS):
            self.mustsets[i] = list(set(tmp_mustsets[i]))
            self.len_mustsets.append(len(self.mustsets[i]))

        self.normalized_size = [0] * self.MS
        min_size = min(self.len_mustsets)
        max_size = max(self.len_mustsets)
        k = (10-1)/(max_size-min_size) # 线性映射到[1,10]
        for i in range(self.MS):
            self.normalized_size[i] = 1 + k * (self.len_mustsets[i] - min_size)

    def InitializeRelatedWord(self,train,train_lab,test,test_lab,Att_List):
        # MS * len(mset)
        self.relatedword = [[[] for y in range(len(self.mustsets[x]))] for x in range(self.MS)]
        self.frequency = [[0 for y in range(len(self.mustsets[x]))] for x in range(self.MS)]
        self.coherence = [[[0 for y in range(self.len_mustsets[x])] for l in range(self.len_mustsets[x])] for x in range(self.MS)]
        self.coherence_value = [[[0 for y in range(self.len_mustsets[x])] for l in range(self.len_mustsets[x])] for x in range(self.MS)]

        self.num_doc_mustsets = [0] * self.MS

        for i in range(len(train)):
            ms = list(train_lab[i]).index(1)
            self.num_doc_mustsets[ms] += 1
            text_line = train[i]
            AttScore = Att_List[i]

            # 统计每个词在每个label下出现的次数
            len_text = len(text_line)
            text_line_set = list(set(text_line)) # 清除重复的word
            len_text_set = len(text_line_set)
            for j in range(len_text_set):
                w1 = text_line_set[j]
                wordidxforset_w1 = self.mustsets[ms].index(w1)
                self.frequency[ms][wordidxforset_w1] += 1
                for k in range(j, len_text_set):
                    w2 = text_line_set[k]
                    wordidxforset_w2 = self.mustsets[ms].index(w2)
                    self.coherence[ms][wordidxforset_w1][wordidxforset_w2] += 1
                    self.coherence[ms][wordidxforset_w2][wordidxforset_w1] += 1

            for j in range(len_text):
                w1 = text_line[j]
                wordidxforset_w1 = self.mustsets[ms].index(w1)
                for l in range(j-self.r, j+self.r+1):
                    if(-1 < l < len_text):
                        if(AttScore[l]!=0.0):
                            w2 = text_line[l]
                            wordidxforset_w2 = self.mustsets[ms].index(w2)
                            self.relatedword[ms][wordidxforset_w1].append([wordidxforset_w2, AttScore[l]])
                            # 清除重复的relatedword

        for i in range(len(test)):
            text_line = test[i]
            AttScore = Att_List[i]
            len_text = len(text_line)
            onehot_lab = test_lab[i]
            mustsetList = [idx for idx, x in enumerate(onehot_lab) if x == 1]

            for ms in mustsetList:
                self.num_doc_mustsets[ms] += 1

            # 统计每个词在每个label下出现的次数
            text_line_set = list(set(text_line)) # 清除重复的word
            len_text_set = len(text_line_set)

            for j in range(len_text_set):
                w1 = text_line_set[j]
                for ms in mustsetList:
                    wordidxforset_w1 = self.mustsets[ms].index(w1)
                    self.frequency[ms][wordidxforset_w1] += 1
                    for k in range(j, len_text_set):
                        w2 = text_line_set[k]
                        wordidxforset_w2 = self.mustsets[ms].index(w2)
                        self.coherence[ms][wordidxforset_w1][wordidxforset_w2] += 1
                        self.coherence[ms][wordidxforset_w2][wordidxforset_w1] += 1

            for j in range(len_text):
                w1 = text_line[j]
                for ms in mustsetList:
                    wordidxforset_w1 = self.mustsets[ms].index(w1)
                    # self.frequency[ms][wordidxforset_w1] += 1
                for l in range(j-self.r, j+self.r+1): # 以j为中心取词
                    if(-1 < l < len_text):
                        if(AttScore[l]!=0.0):
                            w2 = text_line[l]
                            for ms in mustsetList:
                                wordidxforset_w1 = self.mustsets[ms].index(w1)
                                wordidxforset_w2 = self.mustsets[ms].index(w2)
                                w2_idx_list = [pair[0] for pair in self.relatedword[ms][wordidxforset_w1]]
                                if(wordidxforset_w2 in w2_idx_list):
                                    idx_w2 = w2_idx_list.index(wordidxforset_w2)
                                    self.relatedword[ms][wordidxforset_w1][idx_w2][1] = 1/2*(self.relatedword[ms][wordidxforset_w1][idx_w2][1] + AttScore[l]) # 重复的词取平均AttentionScore
                                else:
                                    self.relatedword[ms][wordidxforset_w1].append([wordidxforset_w2, AttScore[l]])

                                # todo: 会发生某个词的realtedword为空
                                # beta [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                                # text [1413, 667, 171, 2524, 1219, 43]

        for ms in range(self.MS):
            D = self.num_doc_mustsets[ms]
            for i in range(self.len_mustsets[ms]):
                D_v = self.frequency[ms][i]
                for j in range(self.len_mustsets[ms]):
                    if(i==j):
                        self.coherence[ms][i][j] = D_v
                    # print(D, D_v, self.coherence[ms][i][j])
                    self.coherence_value[ms][i][j] = (np.math.log(D/D_v))*self.coherence[ms][i][j]
                    # print(self.coherence[ms][i][j])

        # 过滤beta值较小的
        for i in range(self.MS):
            len_set = self.len_mustsets[i]
            for j in range(len_set):
                self.relatedword[i][j].sort(key=lambda x: x[0], reverse=False)
                self.relatedword[i][j].sort(key=lambda x: x[1], reverse=True)
                self.relatedword[i][j][:] = [[x, y] for [x, y] in self.relatedword[i][j] if y > self.threshold]
                # print(self.relatedword[i][j])








