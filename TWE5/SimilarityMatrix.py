"""
 @Time    : 2019/8/31 9:17
 @Author  : Pangjy
 @File    : TWE4
 @Software: PyCharm
 Copy from Evluation/new_tsne, try to caluclate a matrix describing the similiarty between word embedding
"""
import gensim
from gensim.corpora import Dictionary


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def getsimwords(embfile,N,indexoftopic): # 计算余弦相似度
    x = cPickle.load(open(embfile, 'rb'))
    wordemb = x[0]
    topicemb = x[1]
    del x
    topicsimword = {}
    it = 0;

    for i in indexoftopic: # 对于每个在index中的topic
        sim_list = []
        ix = 0
        sum = 0
        for wemb in wordemb: # 对于每个word
            cossim = cos_sim(topicemb[i],wemb) # 计算余弦相似度
            sim_list.append((ix, cossim))
            ix += 1
        sim_list.sort(key = lambda i: i[1], reverse = True) # 排序
        # for item in sim_list[:N]:
        #     sum += item[1];
        # print(sum/100)

        # threshold = 0
        # thresholdix = 0
        # for j in range(len(sim_list)):
        #     if (sim_list[j][1] > 0.7):
        #         threshold = sim_list[j][1]
        #         thresholdix = sim_list[j][0]

    for i in indexoftopic:  # 对于每个topic
        sim_list2 = []
        ix = 0
        for wemb in wordemb:  # 对于每个word
            cosinesim = cos_sim(topicemb[i], wemb)  # 计算余弦相似度
            ixinput = ix
            # if (cosinesim < 0.7):
            #     cosinesim = threshold
            #     ixinput = thresholdix
            sim_list2.append((ixinput, cosinesim))
            ix += 1
        sim_list2.sort(key=lambda i: i[1], reverse=True)  # 排序
        topicsimword[it] = [w[0] for w in sim_list2[:N]]  # 每个topic的
        print(sim_list2[:N])
        # topicsimword[it] = [w[0] for w in sim_list[:N]] # 每个topic的最近的100个
        # print(sim_list[:N])
        it += 1
    return topicsimword

import pandas as pd
import pickle as cPickle
dataset = "Tweet"
if(dataset=="Tweet"):
    docs_count = 2773

def test():
    model = gensim.models.KeyedVectors.load_word2vec_format('../data/GoogleNews-vectors-negative300.bin', binary=True)
    loadpath = '../data/TACL-datasets/langdetect_tweet.p'
    x_data = cPickle.load(open(loadpath, "rb"))

    list1 = [u'apple']
    list2 = [u'pine']
    list3 = [u'pineapple']
    list_sim1 = model.n_similarity(list1, list2)
    print(list_sim1)
    # list_sim2 = model.n_similarity(list2, list3)
    # print(list_sim2)


if __name__ == '__main__':
    test()