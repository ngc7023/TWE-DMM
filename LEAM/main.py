# -*- coding: utf-8 -*-
"""
Guoyin Wang

LEAM
"""

import os, sys
import pickle as cPickle
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import scipy.io as sio
from math import floor
import math
import multiprocessing

from LEAM.model import *
from LEAM.utils import get_minibatches_idx, restore_from_save, tensors_key_in_file, prepare_data_for_emb, load_class_embedding
import cProfile
class Options(object):
    def __init__(self):
        self.GPUID = 0
        self.dataset = 'yelp_full'
        self.fix_emb = True
        self.restore = False
        self.W_emb = 64 # default_none
        self.W_class_emb = 64 # default_none
        self.maxlen = 305
        self.n_words = 1
        self.embed_size = 300
        self.lr = 1e-3
        self.batch_size = 100
        self.max_epochs = 20
        self.dropout = 0.5
        self.part_data = False
        self.portion = 1.0
        self.save_path = "./save/"
        self.log_path = "./log/"
        self.print_freq = 100
        self.valid_freq = 100

        self.optimizer = 'Adam'
        self.clip_grad = None
        self.class_penalty = 1.0
        self.ngram = 55
        self.H_dis = 300


    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value

def emb_classifier(x, x_mask, y, dropout, opt, class_penalty):
    # comment notation
    #  b: batch size, s: sequence length, e: embedding dim, c : num of class
    x_emb, W_norm = embedding(x, opt)  #  b * s * e
    x_emb=tf.cast(x_emb,tf.float32)
    W_norm=tf.cast(W_norm,tf.float32)
    y_pos = tf.argmax(y, -1)
    y_emb, W_class = embedding_class(y_pos, opt, 'class_emb') # b * e, c * e
    y_emb=tf.cast(y_emb,tf.float32)
    W_class=tf.cast(W_class,tf.float32)
    W_class_tran = tf.transpose(W_class, [1,0]) # e * c
    x_emb = tf.expand_dims(x_emb, 3)  # b * s * e * 1
    H_enc = att_emb_ngram_encoder_maxout(x_emb, x_mask, W_class, W_class_tran, opt)
    H_enc = tf.squeeze(H_enc)
    # H_enc=tf.cast(H_enc,tf.float32)
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=False)  # b * c
    logits_class = discriminator_2layer(W_class, opt, dropout, prefix='classify_', num_outputs=opt.num_class, is_reuse=True)
    prob = tf.nn.softmax(logits)
    class_y = tf.constant(name='class_y', shape=[opt.num_class, opt.num_class], dtype=tf.float32, value=np.identity(opt.num_class),)
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)) + class_penalty * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=class_y, logits=logits_class))

    global_step = tf.Variable(0, trainable=False)
    train_op = layers.optimize_loss(
        loss,
        global_step=global_step,
        optimizer=opt.optimizer,
        learning_rate=opt.lr)

    return accuracy, loss, train_op, W_norm, global_step

# def emb_classifier1(x, y, dropout, opt, class_penalty):
    # W_class = embedding_class1(opt, 'class_emb')    # c * e # Topic Embedding 初始化 uniform，否则用relu更新

def emb_classifier2(x, y,gamma, dropout, opt, class_penalty): #190727 add gamma
    # print(opt.topic_distribution)
    #  comment notation
    #  x: 输入训练文本单词id batch_size行，maxlen列
    #  y: 标签数据 一维数组，长度是batch_size
    #  b: batch size, s: sequence length, e: embedding dim, c : num of class
    x_emb, W_norm = embedding(x, opt)  # b * s * e   # x_emb：样本词向量，W_norm: 所有单词embedding
    x_emb = tf.cast(x_emb, tf.float32)
    W_norm = tf.cast(W_norm, tf.float32)
    #W_class = embedding_class1(opt,'class_emb')
    W_class = embedding_class2(opt, 'class_emb')    # c * e # Topic Embedding 初始化 uniform，否则用relu更新
    W_class = tf.cast(W_class, tf.float32)          # W_Class: [opt.num_class * opt.embed_size] = 主题数量 * emb长度
    W_class_tran = tf.transpose(W_class, [1, 0])    # e * c   # tf.transpose：交换张量不同维度，二维相当于转置
    x_emb = tf.expand_dims(x_emb, 3)                # b * s * e * 1 # tf.expand_dims：指定位置增加维度
    H_enc, Att_score,G,Gamma = att_emb_ngram_encoder_maxout1(x_emb, W_class, W_class_tran, opt,gamma)  # H_enc: 文本表示z  #b * 1 * e
    H_enc = tf.squeeze(H_enc)                       # b * e  通过注意力机制将样本文档加权求和得到文本表示H_enc # tf.squeeze：将原始张量中所有维度为1的那些维都删掉的结果

    # 构建判别器，全连接层+线性层 ：输入文档表示H，返回下一个单词的概率 （b * word_num）
    logits = discriminator_2layer(H_enc, opt, dropout, prefix='classify_', num_outputs=opt.n_words,
                                  is_reuse=False)  # loss的计算在下面 #  logits shape=(30, 7042) softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))
    y = tf.cast(y,tf.int64)
    prob_y = tf.nn.softmax(logits)                  # b * word_num
    correct_prediction = tf.equal(tf.argmax(prob_y, 1), y)  # tf.equal：对比这两个张量相等的元素，如果相等返回True，否则返回False
    # argmax返回的是prob_y中的最大值的索引号
    # correct_prediction shape = (30, )
    # y是一个one-hot向量，用来表示下一个单词
    #accuracy = tf.cast(correct_prediction,tf.float32)
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
    return accuracy, loss, train_op, W_norm, W_class, global_step, prob_y, Att_score,G,Gamma

def main():
    # Prepare training and testing data
    opt = Options()
    # load data
    if opt.dataset == 'yahoo':
        loadpath = "./data/yahoo.p"
        embpath = "./data/yahoo_glove.p"
        opt.num_class = 10
        opt.class_name = ['Society Culture',
            'Science Mathematics',
            'Health' ,
            'Education Reference' ,
            'Computers Internet' ,
            'Sports' ,
            'Business Finance' ,
            'Entertainment Music' ,
            'Family Relationships' ,
            'Politics Government']
    elif opt.dataset == 'agnews':
        loadpath = "./data/ag_news.p"
        embpath = "./data/ag_news_glove.p"
        opt.num_class = 4
        opt.class_name = ['World',
                        'Sports',
                        'Business',
                        'Science']
    elif opt.dataset == 'dbpedia':
        loadpath = "./data/dbpedia.p"
        embpath = "./data/dbpedia_glove.p"
        opt.num_class = 14
        opt.class_name = ['Company',
            'Educational Institution',
            'Artist',
            'Athlete',
            'Office Holder',
            'Mean Of Transportation',
            'Building',
            'Natural Place',
            'Village',
            'Animal',
            'Plant',
            'Album',
            'Film',
            'Written Work',
            ]
    elif opt.dataset == 'yelp_full':
        loadpath = "./data/yelp_full.p"
        embpath = "./data/yelp_full_glove.p"
        opt.num_class = 5
        opt.class_name = ['worst',
                        'bad',
                        'middle',
                        'good',
                        'best']
    x = cPickle.load(open(loadpath, "rb"))
    train, val, test = x[0], x[1], x[2]
    train_lab, val_lab, test_lab = x[3], x[4], x[5]
    wordtoix, ixtoword = x[6], x[7]
    del x
    print("load data finished")

    train_lab = np.array(train_lab, dtype='float32')
    val_lab = np.array(val_lab, dtype='float32')
    test_lab = np.array(test_lab, dtype='float32')
    opt.n_words = len(ixtoword)
    if opt.part_data:
        #np.random.seed(123)
        train_ind = np.random.choice(len(train), int(len(train)*opt.portion), replace=False)
        train = [train[t] for t in train_ind]
        train_lab = [train_lab[t] for t in train_ind]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.GPUID)

    print(dict(opt))
    print('Total words: %d' % opt.n_words)

    try:
        opt.W_emb = np.array(cPickle.load(open(embpath, 'rb')),dtype='float32')
        opt.W_class_emb =  load_class_embedding( wordtoix, opt)
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:1'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen],name='x_')
        x_mask_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.maxlen],name='x_mask_')
        keep_prob = tf.placeholder(tf.float32,name='keep_prob')
        y_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.num_class],name='y_')
        class_penalty_ = tf.placeholder(tf.float32, shape=())
        accuracy_, loss_, train_op, W_norm_, global_step = emb_classifier(x_, x_mask_, y_, keep_prob, opt, class_penalty_)
    uidx = 0
    max_val_accuracy = 0.
    max_test_accuracy = 0.

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
                kf = get_minibatches_idx(len(train), opt.batch_size, shuffle=True)
                for _, train_index in kf:
                    uidx += 1
                    sents = [train[t] for t in train_index]
                    x_labels = [train_lab[t] for t in train_index]
                    x_labels = np.array(x_labels)
                    x_labels = x_labels.reshape((len(x_labels), opt.num_class))

                    x_batch, x_batch_mask = prepare_data_for_emb(sents, opt)
                    _, loss, step,  = sess.run([train_op, loss_, global_step], feed_dict={x_: x_batch, x_mask_: x_batch_mask, y_: x_labels, keep_prob: opt.dropout, class_penalty_:opt.class_penalty})

                    if uidx % opt.valid_freq == 0:
                        train_correct = 0.0
                        # sample evaluate accuaccy on 500 sample data
                        kf_train = get_minibatches_idx(500, opt.batch_size, shuffle=True)
                        for _, train_index in kf_train:
                            train_sents = [train[t] for t in train_index]
                            train_labels = [train_lab[t] for t in train_index]
                            train_labels = np.array(train_labels)
                            train_labels = train_labels.reshape((len(train_labels), opt.num_class))
                            x_train_batch, x_train_batch_mask = prepare_data_for_emb(train_sents, opt)
                            train_accuracy = sess.run(accuracy_, feed_dict={x_: x_train_batch, x_mask_: x_train_batch_mask, y_: train_labels, keep_prob: 1.0, class_penalty_:0.0})

                            train_correct += train_accuracy * len(train_index)

                        train_accuracy = train_correct / 500

                        print("Iteration %d: Training loss %f " % (uidx, loss))
                        print("Train accuracy %f " % train_accuracy)

                        val_correct = 0.0
                        kf_val = get_minibatches_idx(len(val), opt.batch_size, shuffle=True)
                        for _, val_index in kf_val:
                            val_sents = [val[t] for t in val_index]
                            val_labels = [val_lab[t] for t in val_index]
                            val_labels = np.array(val_labels)
                            val_labels = val_labels.reshape((len(val_labels), opt.num_class))
                            x_val_batch, x_val_batch_mask = prepare_data_for_emb(val_sents, opt)
                            val_accuracy = sess.run(accuracy_, feed_dict={x_: x_val_batch, x_mask_: x_val_batch_mask,
                                y_: val_labels, keep_prob: 1.0,
                                class_penalty_:0.0                                         })

                            val_correct += val_accuracy * len(val_index)

                        val_accuracy = val_correct / len(val)
                        print("Validation accuracy %f " % val_accuracy)

                        if val_accuracy > max_val_accuracy:
                            max_val_accuracy = val_accuracy

                            test_correct = 0.0

                            kf_test = get_minibatches_idx(len(test), opt.batch_size, shuffle=True)
                            for _, test_index in kf_test:
                                test_sents = [test[t] for t in test_index]
                                test_labels = [test_lab[t] for t in test_index]
                                test_labels = np.array(test_labels)
                                test_labels = test_labels.reshape((len(test_labels), opt.num_class))
                                x_test_batch, x_test_batch_mask = prepare_data_for_emb(test_sents, opt)

                                test_accuracy = sess.run(accuracy_,feed_dict={x_: x_test_batch, x_mask_: x_test_batch_mask,y_: test_labels, keep_prob: 1.0, class_penalty_: 0.0})

                                test_correct += test_accuracy * len(test_index)
                            test_accuracy = test_correct / len(test)
                            print("Test accuracy %f " % test_accuracy)
                            max_test_accuracy = test_accuracy

                print("Epoch %d: Max Test accuracy %f" % (epoch, max_test_accuracy))
                saver.save(sess, opt.save_path, global_step=epoch)
            print("Max Test accuracy %f " % max_test_accuracy)

        except KeyboardInterrupt:
            print('Training interupted')
            print("Max Test accuracy %f " % max_test_accuracy)


from TWE2.data_process import make_train_data, get_train_by_label,get_train_by_label1
import time
#@profile
def Train_TWE(opt,train_lab,dmm): # Option由TWE2/main控制
    # 加载预训练的Word embedding
    try:
        opt.W_emb = np.array(cPickle.load(open(opt.embpath, 'rb'))[0], dtype='float32')
        print("Load pretrained word embedding")
        opt.topic_emb = dmm.initTopicEmb(opt.W_emb)
        print("Load topic embedding")
    except IOError:
        print('No embedding file found.')
        opt.fix_emb = False

    with tf.device('/gpu:0'):
        x_ = tf.placeholder(tf.int32, shape=[opt.batch_size, opt.maxlen], name='x_')  # 输入训练文本单词id batch_size行，maxlen列
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        y_ = tf.placeholder(tf.int32, shape=[opt.batch_size, ], name='y_')  # 标签数据 一维数组，长度是batch_size
        gamma_ = tf.placeholder(tf.float32, shape=[opt.batch_size, opt.num_class], name='gamma_')  # 标签数据 一维数组，长度是batch_size
        class_penalty_ = tf.placeholder(tf.float32, shape=())  # 可能是正则化
        accuracy_, loss_, train_op, W_norm_, W_class_, global_step, prob_, att_score,G,gamma = emb_classifier2(x_, y_,gamma_, keep_prob,
                                                                                                      opt,
                                                                                                      class_penalty_)
    uidx = 0


    config = tf.ConfigProto(
log_device_placement=False,
allow_soft_placement=True,
device_count={"CPU":4},
inter_op_parallelism_threads=4,
intra_op_parallelism_threads=4,)
    config.gpu_options.allow_growth = True
    np.set_printoptions(precision=3)
    np.set_printoptions(threshold=np.inf)
    saver = tf.train.Saver()

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if opt.restore:
            try:
                t_vars = tf.trainable_variables()
                save_keys = tensors_key_in_file(opt.save_path)
                ss = set([var.name for var in t_vars]) & set([s + ":0" for s in save_keys.keys()])
                cc = {var.name: var for var in t_vars}
                ss_right_shape = set([s for s in ss if cc[s].get_shape() == save_keys[s[:-2]]])

                loader = tf.train.Saver(var_list=[var for var in t_vars if var.name in ss_right_shape])
                loader.restore(sess, opt.save_path)

                print("Loading variables from '%s'." % opt.save_path)
                print("Loaded variables:" + str(ss))

            except:
                print("No saving session, using random initialization")
                sess.run(tf.global_variables_initializer())

        try:
            iter_cohere = 0
            for epoch in range(opt.max_epochs):
                print("Starting epoch %d" % epoch)
                print("Prepare data")
                time_start = time.time()
                tmp_prob = [0]*opt.sample_number
                #tmp_prob = []
                #for i in range(opt.sample_number):
                    #tmp_prob.append(0)                  # n_words = None tmp_prob是list,长度为样本长

                kf = get_minibatches_idx(opt.sample_number, opt.batch_size,
                                         shuffle=True)  # zip(range(len(minibatches)), minibatches)
                time_data = 0
                time_net = 0
                avgloss = 0
                avgacc = 0
                iter_time = 0
                for _, train_index in kf:
                    uidx += 1
                    sents = []
                    x_labels = []
                    gamma_sent = []
                    time_data_1 = time.time()
                    for idx in train_index:
                        label = train_lab[idx] # row and column
                        sample_and_label = get_train_by_label(label[1],opt.maxlen,dmm.dpre.docs[label[0]].words) # return maxlen + 1
                        sents.append(sample_and_label[:opt.maxlen])
                        x_labels.append(sample_and_label[opt.maxlen])
                        gamma_sent.append(opt.topic_distribution[label[0]])
                    time_data_2 = time.time()
                    time_data += (time_data_2-time_data_1)
                    time_net_1 = time.time()
                    acc,_, loss, step, prob, word_emb, topic_emb, att_,G_,Gamma_ = sess.run(
                        [accuracy_,train_op, loss_, global_step, prob_, W_norm_, W_class_, att_score,G,gamma],
                        feed_dict={x_: sents, y_: x_labels,gamma_:gamma_sent,
                                   keep_prob: opt.dropout, class_penalty_: opt.class_penalty})
                    time_net_2 = time.time()
                    time_net += (time_net_2 - time_net_1)
                    #print(acc)
                    #exit()
                    #print(G_[0])
                    #print(Gamma_[0])
                    #exit()
                    if math.isnan(loss):
                        print(uidx, loss)
                        return

                    if uidx % opt.valid_freq == 0:
                        time_end = time.time()
                        time_interval = time_end-time_start
                        print("Iteration %d: Training loss %f   Time cost: %f" % (uidx, loss,time_interval))
                        avgloss+=loss
                        avgacc+=acc
                        iter_time +=1
                    ai = 0
                    for t in train_index:
                        tmp_prob[t]=prob[ai][x_labels[ai]] #tmp_prob shape = (sample_number)
                        ai+=1
                saver.save(sess, opt.save_path, global_step=epoch)
                print("average loss:%f average accuracy:%f"%(avgloss/iter_time,avgacc/iter_time)) 
                iter_cohere +=1
                # 进行一次gibbs抽样 已处理过所有样本
                dmm.prob = tmp_prob	
                dmm.est2(opt)
                if(iter_cohere%10==0):
                    dmm._phi()
                    print('Topic coherence:', dmm.getTopicCoherence()) 

            cPickle.dump([word_emb, topic_emb], open(opt.topicwordemb_path, 'wb'))
            dmm.save1()
            print('Topic coherence:', dmm.getTopicCoherence())
            
        except KeyboardInterrupt:
            print('Training interupted')
            dmm.save1()
            print('Topic model is saved!')

if __name__ == '__main__':
    main()
