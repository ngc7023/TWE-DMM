import pandas as pd
class MustSets(object):
    def __init__(self):
        self.mustsets = []
        self.wordstrToMustsetListMap = dict()
        self.length = 0

    def InitMustSets(self,vocab_dict,text,labelfile):
        # 只能处理label已经是顺序的文件，corpus和label是对应的
        df = pd.read_csv(labelfile, names=['label'])
        label_list = df['label'].values.tolist()
        tmp_list = list(set(label_list))
        tmp_list.sort(key=label_list.index)
        label_dic = dict(zip(tmp_list, list(range(0, len(tmp_list)))))

        self.mustsets = [[] for i in range(len(label_dic))]
        tmp_mustsets = [[] for i in range(len(label_dic))]

        idx_lable = 0
        for line in text:
            for key in label_dic.keys():
                if(label_list[idx_lable]==key):
                    tmp_mustsets[label_dic[key]]+=line
                    idx_lable+=1
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


    def getMustSetListGivenWordstr(self,wordstr):
        return self.wordstrToMustsetListMap.get(wordstr)



