import pickle as cPickle
from TWE.utils import *

# loadpath = "../data/news_train_text.p"
# embpath = "../data/train_text_emb.p"
#
# elif opt.dataset == 'classifydata':
loadpath = '../data/classifydata/classifydata_index.p'
embpath = "../data/classifydata/classifydata_emb.p"

# W_emb = np.array(cPickle.load(open(embpath, 'rb'))[0], dtype='float32')
# print(len(W_emb))
# print(W_emb[1])

x = cPickle.load(open(loadpath, "rb"))
train_before = x[0]
train, train_lab = make_train_data(x[0], 30)

wordtoix, ixtoword = x[2], x[3]  #ixtoword/wordtoix都是dict，保存单词-index映射关系
#
print(len(train_before)) # 原样本数
# print(len(train))        # 样本数扩增 将train_before不断向前移动，并且取下一个字作为label
print(len(train_lab))
#
#
print("打印原短文")
for i in range(3505,3506):
    str_bef = ""
    for wordix in train_before[i]:
        word = ixtoword[wordix]
        str_bef += word
    print("第"+str(i)+"条text，长度为"+str(len(train_before[i])))
    print(str_bef)
#
print("打印经窗口滑动后产生的样本")
for i in range(107180,107183):
    str = ""
    for wordix in train[i]:
        word = ixtoword[wordix]
        str += word
    # print(wordix)
    print(str)
    label = ixtoword[train_lab[i]]
    print(label)

# list = [2814, 2400, 7027, 0, 669, 0, 943, 3034, 0, 0, 7028, 2895, 555, 6079, 0, 6151, 0, 0, 0, 0, 7027, 5759, 2697, 3062, 7027, 7029, 0, 669, 15, 1, 0]
# str=""
# for wordix in list:
#     word = ixtoword[wordix]
#     str +=word
# print(str+"\n")
#
# str_bef = ""
# for wordix in train_before[len(train_before)-1]:
#     word = ixtoword[wordix]
#     str_bef += word
# print(str_bef+"\n")
#
# str_bef = ""
# for wordix in train_before[0]:
#     word = ixtoword[wordix]
#     str_bef += word
# print(str_bef)