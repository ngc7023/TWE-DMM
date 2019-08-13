from LLA.LDA import *
from LLA.utils import *
import tensorflow as tf

class Options(object):
	def __init__(self):
		self.GPUID = 0
		self.dataset = 'classifydata'
		self.is_training=True
		self.num_layer=1 # LSTM结构层数
		self.embed_size = 64  # embedding 维度
		self.lr = 1e-3   # 学习率
		self.batch_size = 20 #default_20
		self.train_num_step=35 #数据截断长度
		self.max_epochs = 200 # 迭代论述
		self.num_class = 40  # 主题个数
		self.lstm_keep_prob=0.9 # LSTM节点不被dropout的概率
		self.embedding_keep_prob=0.9 # 词向量不被dropout的概率
		self.save_path = "./save/"
		self.log_path = "./log/"
		self.optimizer = 'Adam'
		self.clip_grad = 5  #梯度大小上限

def main():
	opt = Options()
	# load data
	if opt.dataset == 'train_text':
		loadpath = '../data/news_train_text.p'
		embpath = "../data/train_text_emb.p"
	elif opt.dataset == 'train_title':
		loadpath = '../data/news_train_title.p'
		embpath = "../data/train_title_emb.p"
	elif opt.dataset == 'classifydata':
		loadpath = '../data/classifydata/classifydata_index.p'
		embpath = "../data/classifydata/classifydata_emb.p"
	else:
		pass
	lda=LDAmodel(loadpath,opt.num_class)
	x = cPickle.load(open(loadpath, "rb"))
	train = x[0]
	train_lab = lda.Z
	print(len(train), len(train_lab))
	wordtoix, ixtoword = x[2], x[3]
	del x  # del删除的是变量，而不是数据
	print("load data finished")
	print('sample number: ', len(train))
	train_lab = np.array(train_lab)
	opt.n_words = len(ixtoword) # 词典规模
	print('Total words: %d' % opt.n_words)

	with tf.device('/cpu:0'):
		x_ = tf.placeholder(tf.int32, shape = [opt.batch_size, opt.train_num_step], name = 'x_')  # 输入训练文本单词id
		y_ = tf.placeholder(tf.int32, shape = [opt.batch_size, opt.train_num_step], name = 'y_')  # 标签数据
		dropout_keep_prob=opt.lstm_keep_prob if opt.is_training else 1.0
		lstm_cells=[
			tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(opt.embed_size),
			output_keep_prob=dropout_keep_prob)
			for _ in range(opt.num_layer)]
		cell=tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
		init_state=cell.zero_state(opt.batch_size,tf.float32)
		embedding=tf.get_variable('embedding',[opt.n_words,opt.embed_size])
		inputs=tf.nn.embedding_lookup(embedding,x_)

		if opt.is_training:
			inputs=tf.nn.dropout(inputs,opt.embedding_keep_prob)
		outputs=[]
		state=init_state

		with tf.variable_scope('RNN'):
			for time_step in range(opt.train_num_step):
				if time_step>0:
					tf.get_variable_scope().reuse_variables()
				cell_output,state=cell(inputs[:,time_step,:],state)
				outputs.append(cell_output)
		output=tf.reshape(tf.concat(outputs,1),[-1,opt.embed_size])
		weight=tf.get_variable('weight',[opt.embed_size,opt.num_class])
		bais=tf.get_variable('bais',[opt.num_class])
		logits=tf.matmul(output,weight)+bais
		prob=tf.nn.softmax(logits)

		loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.reshape(y_,[-1]),logits = logits)
		cost=tf.reduce_sum(loss)/opt.batch_size
		final_state=state

		if not opt.is_training:
			return

		trainable_variables=tf.trainable_variables()
		grads,_=tf.clip_by_global_norm(tf.gradients(cost,trainable_variables),opt.clip_grad)
		optimizer=tf.train.AdamOptimizer(learning_rate = opt.lr)
		train_op=optimizer.apply_gradients(zip(grads,trainable_variables))

	uidx = 0
	max_train_accuracy = 0.

	config = tf.ConfigProto(log_device_placement = False, allow_soft_placement = True, )
	config.gpu_options.allow_growth = True
	np.set_printoptions(precision = 3)
	np.set_printoptions(threshold = np.inf)
	saver = tf.train.Saver()

	with tf.Session(config = config) as sess:
		train_writer = tf.summary.FileWriter(opt.log_path + '/train', sess.graph)
		test_writer = tf.summary.FileWriter(opt.log_path + '/test', sess.graph)
		sess.run(tf.global_variables_initializer())

		try:
			step=0
			for epoch in range(opt.max_epochs):
				print("Starting epoch %d" % epoch)
				# 获取embedding训练数据
				sentsdata, sentslabel, total_words = make_batches(train, train_lab, opt.batch_size, opt.train_num_step)
				# sentsdata, sentslabel, total_words = make_batches_topic(train, train_lab, opt.batch_size, opt.train_num_step)
				total_cost=0
				iters=0
				total_prob=[]
				for i in range(len(sentsdata)):
					x_batch=sentsdata[i]
					y_batch=sentslabel[i]
					state=sess.run(init_state)
					cost_,state_,_,prob_=sess.run(
						[cost,final_state,train_op,prob],
						feed_dict = {x_:x_batch,y_:y_batch,init_state:state}
					)
					total_cost+=cost_
					iters+=opt.train_num_step
					total_prob+=list(prob_)

					if step%100==0:
						print('After %d steps: Training loss %f' %(step,total_cost/iters))
					step+=1
				saver.save(sess, opt.save_path, global_step = epoch)

				# 进行一次gibbs抽样
				lda.topicprob=total_prob
				lda.est()
				train_lab =lda.Z
				if((epoch+1)%10==0):
					lda._phi()
					print("topic coherence:",lda.getTopicCoherence())
			lda.save()

		except KeyboardInterrupt:
			print('Training interupted')
			print("Max Train accuracy %f " % max_train_accuracy)
			lda.save()
			print('Topic model is saved!')


if __name__ == '__main__':
	main()

