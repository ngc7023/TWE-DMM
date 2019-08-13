# -*- coding: utf-8 -*-
# 2019.07.15 修改label数据结构
import os, sys
import _pickle as cPickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from math import floor

from TWE.new_DMM import *
from TWE.model import *
from TWE.new_utils import *
import time


class Options(object):
    def __init__(self):
        self.GPUID = 0
        self.dataset = 'classifydata'  # default 'classifydata'
        self.fix_emb = True  # Word embedding 初始化方式判断
        self.restore = False
        self.W_emb = None  # Word embedding初始化矩阵
        self.W_class_emb = None  # label embedding 初始化矩阵
        self.maxlen = 30  # 序列最大长度
        self.n_words = None
        self.embed_size = 64  # embedding 维度
        self.lr = 1e-3  # 学习率
        self.batch_size = 128  # default_30
        self.max_epochs = 1  # default_200
        self.num_class = 40  # 主题个数
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0
        self.save_path = "./save/"
        self.log_path = "./log/"
        self.print_freq = 100
        self.valid_freq = 100

        self.optimizer = 'Adam'
        self.clip_grad = 5
        self.class_penalty = 1.0
        self.ngram = 1.0
        self.H_dis = 64  # 全连接层单元个数，相当于输出维度？？？？？

        self.ixtoword = None
        self.wordtoix = None
        self.label = None

    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


def emb_classifier(x, y, dropout, opt, class_penalty):
    #  comment notation
    #  x: 输入训练文本单词id batch_size行，maxlen列
    #  y: 标签数据 一维数组，长度是batch_size
    #  b: batch size, s: sequence length, e: embedding dim, c : num of class
    x_emb, W_norm = embedding(x, opt)  # b * s * e   # x_emb：样本词向量，W_norm: 所有单词embedding
    # x_emb (batch_size * 序列 maxlen * embed长度) (30 * 30 * 64)
    # W_norm (n_words * embed长度）（7042 * 64）
    x_emb = tf.cast(x_emb, tf.float32)  # tf.cast：转化数据格式为指定类型
    W_norm = tf.cast(W_norm, tf.float32)
    # w_emb = tf.nn.embedding_lookup(W_norm, w)  # 预测单词的embedding
    # y_pos = tf.argmax(y, -1)  # 返回的是vector中的最大值的索引号
    W_class = embedding_class(opt, 'class_emb')  # c * e # Topic Embedding 初始化 uniform，否则用relu更新
    W_class = tf.cast(W_class, tf.float32)  # W_Class: [opt.num_class * opt.embed_size] = 主题数量 * emb长度
    W_class_tran = tf.transpose(W_class, [1, 0])  # e * c   # tf.transpose：交换张量不同维度，二维相当于转置
    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1 # tf.expand_dims：指定位置增加维度
    H_enc, Att_score = att_emb_ngram_encoder_maxout(x_emb, W_class, W_class_tran, opt)  # H_enc: 文本表示z  #b * 1 * e
    H_enc = tf.squeeze(H_enc)  # b * e  通过注意力机制将样本文档加权求和得到文本表示H_enc # tf.squeeze：将原始张量中所有维度为1的那些维都删掉的结果
    # H_enc=tf.cast(H_enc,tf.float32)
    # 构建判别器，全连接层+线性层 ：输入文档表示H，返回下一个单词的概率 （b * word_count）
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.n_words,
                                  is_reuse=False)  # loss的计算在下面 #  logits shape=(30, 7042) softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
    # logits_class = discriminator_2layer(W_class, opt, dropout, prefix = 'classify_', num_outputs = opt.num_class,
    # is_reuse = True)
    # logits_word=discriminator_2layer(W_norm, opt, dropout, prefix = 'wordclassify_', num_outputs = opt.n_words,
    # 									is_reuse = False)

    prob_y = tf.nn.softmax(logits)  # 返回 30 * 7042的概率
    # x_pos=tf.one_hot(tf.squeeze(x),opt.n_words)
    # prob_w =tf.reduce_max(prob,axis = -1)   #??????????
    class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32,
                          value=np.identity(opt.num_class), )
    correct_prediction = tf.equal(tf.argmax(prob_y, 1), tf.argmax(y, 1))  # tf.equal：对比这两个张量相等的元素，如果相等返回True，否则返回False
    # argmax返回的是prob_y中的最大值的索引号
    # correct_prediction shape = (30, )
    # y是一个one-hot向量，用来表示下一个单词
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 交叉熵计算损失
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.squeeze(y), logits=logits))
    loss = loss / opt.batch_size

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opt.optimizer,
        learning_rate=opt.lr)

    # trainable_variables=tf.trainable_variables()
    # grads,_=tf.clip_by_global_norm(tf.gradients(loss,trainable_variables),opt.clip_grad)
    # optimizer=tf.train.AdamOptimizer(opt.lr)
    # train_op=optimizer.apply_gradients(zip(grads,trainable_variables))
    return accuracy, loss, train_op, W_norm, W_class, global_step, prob_y, Att_score

def main():
    # Prepare training and testing data
    opt = Options()
    # load data
    if opt.dataset == 'train_text':
        # loadpath = "../data/train_text_corpus.p"
        loadpath = '../data/news_train_text.p'
        embpath = "../data/train_text_emb.p"
    # dmm = DMMmodel('../data/news_train_text.p')
    elif opt.dataset == 'train_title':
        # loadpath = "../data/train_title_corpus.p"
        loadpath = '../data/news_train_title.p'
        embpath = "../data/train_title_emb.p"
    # dmm = DMMmodel('../data/news_train_title.p')
    elif opt.dataset == 'classifydata':
        loadpath = '../data/classifydata/classifydata_index.p'
        embpath = "../data/classifydata/classifydata_emb.p"
    # dmm = DMMmodel('../data/classifydata/classifydata_index.p')
    else:
        pass

    dmm = DMMmodel(loadpath, opt.num_class)
    print("docs_count",dmm.dpre.docs_count)
    doctopic_initpath = '../RE/DMM/k_' + str(opt.num_class) + '/testDMM.topicAssignments'
    x = cPickle.load(open(loadpath, "rb"))
    train_lab, sample_number = make_train_data(x[0])

    allwordseq = x[0]
    wordtoix, ixtoword = x[2], x[3]  # ixtoword/wordtoix都是dict，保存单词-index映射关系
    del x  # del删除的是变量，而不是数据

    dmm.init_Z(dmm.Z)

    print("load data finished")
    print('sample number: ', sample_number)

    train_lab = np.array(train_lab, dtype='float32')
    opt.n_words = len(ixtoword)
    opt.wordtoix = wordtoix
    opt.ixtoword = ixtoword
    del wordtoix, ixtoword

    # if opt.part_data:  # 是否使用部分数据（dropout使用？） #default_False
    # 	train_ind = np.random.choice(len(train), int(len(train) * opt.portion), replace = False)
    # 	train = [train[t] for t in train_ind]
    # 	train_lab = [train_lab[t] for t in train_ind]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPUID)

    # print(dict(opt))
    print('Total words: %d' % opt.n_words)

    try:
        opt.W_emb = np.array(cPickle.load(open(embpath, 'rb'))[0], dtype='float32')  # 加载预训练的Word embedding
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/cpu:0'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen], name='x_')  # 输入训练文本单词id batch_size行，maxlen列
        # x_mask_ = tf.placeholder(tf.float32, shape = [opt.batch_size, opt.maxlen], name = 'x_mask_')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        y_ = tf.placeholder(tf.int32, shape=[opt.batch_size, ], name='y_')  # 标签数据 一维数组，长度是batch_size
        # w_=tf.placeholder(tf.int32, shape = [opt.batch_size, ], name = 'w_')
        class_penalty_ = tf.placeholder(tf.float32, shape=())  # 可能是正则化

        accuracy_, loss_, train_op, W_norm_, W_class_, global_step, prob_, att_score = emb_classifier(x_, y_, keep_prob,
                                                                                                      opt,
                                                                                                      class_penalty_)
    uidx = 0
    max_train_accuracy = 0.

    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, )
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                t_vars = tf.trainable_variables()
                save_keys = tensors_key_in_file(opt.save_path)
                ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
                cc = {var.name: var for var in t_vars}
                # only restore variables with correct shape
                ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

                loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
                loader.restore(sess, opt.save_path)

                print("Loading variables from '%s'." % opt.save_path)
                print("Loaded variables:" + str(ss))

            except:
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        try:
            for epoch in range(opt.max_epochs):
                print("Starting epoch %d" % epoch)
                time_start = time.time()
                time_sum = 0
                # 获取embedding训练数据
                # attention=[]
                # for it in range(len(t(rain)):
                # 	attention.append([0 ]*len(train[it]))
                tmp_prob = []
                for i in range(sample_number):
                    tmp_prob.append(0)  # n_words = None tmp_prob是list,长度为样本长

                kf = get_minibatches_idx(sample_number, opt.batch_size,
                                         shuffle=True)  # zip(range(len(minibatches)), minibatches)
                for _, train_index in kf:
                    uidx += 1
                    sents = []
                    x_labels = []

                    for q in range(len(train_index)): # train_index[t] = 352153
                        label = train_lab[train_index[q]]
                        row = label[0].astype(np.int)
                        column = label[1].astype(np.int)
                        sents.append(get_train_by_label(label,opt.maxlen,allwordseq[row]))
                        x_labels.append(allwordseq[row][column])
                    # sents = [train[t] for t in train_index] # 30*30 wordix
                    # x_labels = [train_lab[t] for t in train_index]
                    _, loss, step, prob, word_emb, topic_emb, att_ = sess.run(
                        [train_op, loss_, global_step, prob_, W_norm_, W_class_, att_score],
                        feed_dict={x_: sents, y_: x_labels,
                                   keep_prob: opt.dropout, class_penalty_: opt.class_penalty})

                    if math.isnan(loss):
                        print(uidx, loss)
                        return

                    if uidx % opt.valid_freq == 0:
                        time_end = time.time()
                        time_interval = time_end - time_start
                        time_sum += time_interval
                        time_start = time.time()
                        print("Iteration %d: Training loss %f   Time Interval: %f" % (uidx, loss, time_interval))

                    ai = 0
                    for t in train_index:
                        tmp_prob[t]=prob[ai][x_labels[ai]] #tmp_prob shape = (sample_number)
                        ai+=1
                    # for t in train_index: # 只会更新下标为t的tmp_prob
                        # # attention[t]=att_[ai]
                        # tmp_prob[t] = prob[ai]  # prob = prob_ = embclassifier的prob_y
                        # ai += 1

                # print(len(tmp_prob))
                saver.save(sess, opt.save_path, global_step=epoch)
                time_start2 = time.time()
                # 进行一次gibbs抽样 已处理过所有样本
                # dmm.prob=prob
                # for prob in tmp_prob:
                #     print(prob)
                dmm.prob = tmp_prob
                dmm.est()  # 为每篇文档分配主题
                time_end2 = time.time()
                time_interval2 = time_end2-time_start2
                print("Time for DMM: ", time_interval2)
                time_sum+=time_interval2
            # train_lab = dmm.Z
            dmm.save()  # 保存doc-topic，topic-word分布，每个文档的主题，topNword

            cPickle.dump([word_emb, topic_emb], open('./re/topic-wordemb.p', 'wb'))
            # cPickle.dump([attention],open('./re/attention.p','wb'))
            print('topic coherence:', dmm.getTopicCoherence())
            print('time_sum:', time_sum)
            # print("Max Train accuracy %f " % max_train_accuracy)

        except KeyboardInterrupt:
            print('Training interupted')
            print("Max Train accuracy %f " % max_train_accuracy)
            dmm.save()
            print('Topic model is saved!')


if __name__ == '__main__':
    main()
