"""
author: Pangjy
reference:
"""
import _pickle as cPickle
import numpy as np
import codecs
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


def loademb0(embfile):
    x = cPickle.load(open(embfile, 'rb'))
    wordemb = x[0]
    topicemb = x[1]
    del x
    return wordemb, topicemb

def loademb(embfile,topic_words,indexoftopic):
    x=cPickle.load(open(embfile,'rb'))
    wordemb=x[0]
    topicemb0=x[1];topicemb=[];
    for i in indexoftopic:
        topicemb.append(topicemb0[i]);
    del x

    wordemb_=[]
    target_w_=[]
    for k in range(len(topic_words)): # 5*100 len（topic_words）= topic_number
        for wi in topic_words[k]:
            wordemb_.append(wordemb[wi])
            target_w_.append(k) # 小圆颜色
    target_t=[k for k in range(len(topic_words))]; #大圆颜色
    topicwordemb=wordemb_+list(topicemb);
    X_tsne=TSNE(learning_rate = 100).fit_transform(topicwordemb)
    X_word=X_tsne[:len(wordemb_)]
    X_topic=X_tsne[len(wordemb_):]

    fig, _ = plt.subplots()
    plt.subplot(111)
    plt.scatter(X_word[:, 0], X_word[:, 1], marker='.',s=20, c = target_w_)
    plt.scatter(X_topic[:, 0], X_topic[:, 1], marker = 'o',s=60, c = target_t,edgecolors = 'k')
    plt.title('t-SNE plot of topic word embedding')
    plt.xticks([])
    plt.yticks([])

    plt.show()

def loadphi(phifile):
    f = codecs.open(phifile, 'r')
    phi = []
    for line in f:
        tokens = line.strip().split()
        prob = [float(p) for p in tokens]
        twords = [(n, prob[n]) for n in range(len(prob))]
        twords.sort(key=lambda i: i[1], reverse=True)
        phi.append(twords)
    return phi


def topic_emb_value(wordemb, topicemb, phi):
    re = []
    for i in range(len(phi)):  # 打印所有数据
        print('topic:', i)
        te = topicemb[i]  # topic-embedding
        value = []
        count = 0
        tmp = np.zeros(len(topicemb[0]))
        for w in phi[i]:
            tmp += wordemb[w[0]]
            count += 1
            if count % 20 == 0:
                value.append(cos_sim(te, tmp))
            if count >= 100:
                break
        re.append(value)
        print(value)
    return re


def plot_topic_value(re):
    re = np.array(re)
    re = re.T

    indexoftopic = []

    y = sorted(re[4], reverse=True)  # 'Top-100' 已排序
    print(y)
    for i in range(len(y)):
        indexoftopic.append(re[4].tolist().index(y[i])); # 找下标，也就是topic
    print(indexoftopic)

    return indexoftopic

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

if __name__ == '__main__':
    embfile = '../RE/TWE_DMM/k_10/topic-wordemb.p'
    phifile = '../RE/TWE_DMM/k_10/phifile.txt'
    wordemb, topicemb = loademb0(embfile)
    phi = loadphi(phifile)
    re = topic_emb_value(wordemb, topicemb, phi)
    indexoftopic = plot_topic_value(re) # 找到Top100的similarity已排序的topic下标
    indexoftopic = [9,8,7,6,5,4,3,2,1,0]
    # indexoftopic = indexoftopic[:10]
    # indexoftopic.reverse()
    print(indexoftopic)

    topic_words = getsimwords(embfile, 100, indexoftopic)  # 10*100 为部分topic找他们最相近的word 100是top-100
    loademb(embfile, topic_words,indexoftopic)



