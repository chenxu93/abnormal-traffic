#!/usr/bin/python
#-*- encoding:utf-8 -*-


import tensorflow as tf 
import numpy as np 
from inpudata import DataSet
import time
import os
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.metrics import confusion_matrix

#============================================================================
#reading data

def data_prepare(f1_name,f2_name,y1,y2):
	d1 = f1_name.values
	d2 = f2_name.values
	d1[:,-1] = y1
	d2[:,-1] = y2

	dataset = np.concatenate((d1,d2),axis=0)

	#打乱
	np.random.shuffle(dataset)
	return dataset

def data2feature(f_name,cla):
	file_value = f_name.values
	file_value[:,-1] = cla
	feature = file_value
	feature = feature[:,1:]
	np.random.shuffle(feature)
	return feature

def discard_fiv_tupple(data):
	
	for i in range(10):
		#protoc
		data[:,7+i*160] = 0
		#ip and port
		data[:,10+i*160:22+i*160] = 0
	return data


start = time.time()

Benign = pd.read_csv("../flow_labeled/labeld_Monday-Benign.csv")#339621

DoS_GoldenEye = pd.read_csv("../flow_labeled/labeld_DoS-GlodenEye.csv")#7458

# Heartbleed = pd.read_csv("../flow_labeled/labeld_Heartbleed-Port.csv")#1

DoS_Hulk = pd.read_csv("../flow_labeled/labeld_DoS-Hulk.csv")#14108

DoS_Slowhttps = pd.read_csv("../flow_labeled/labeld_DoS-Slowhttptest.csv")#4216

DoS_Slowloris = pd.read_csv("../flow_labeled/labeld_DoS-Slowloris.csv")#3869

SSH_Patator = pd.read_csv("../flow_labeled/labeld_SSH-Patator.csv")#2511

FTP_Patator = pd.read_csv("../flow_labeled/labeld_FTP-Patator.csv")#3907

Web_Attack_BruteForce = pd.read_csv("../flow_labeled/labeld_WebAttack-BruteForce.csv")#1353
Web_Attack_SqlInjection = pd.read_csv("../flow_labeled/labeld_WebAttack-SqlInjection.csv")#12
Web_Attack_XSS = pd.read_csv("../flow_labeled/labeld_WebAttack-XSS.csv")#631

# Infiltraton = pd.read_csv()#3

Botnet = pd.read_csv("../flow_labeled/labeld_Botnet.csv")#1441

PortScan_1 = pd.read_csv("../flow_labeled/labeld_PortScan_1.csv")#344
PortScan_2 = pd.read_csv("../flow_labeled/labeld_PortScan_2.csv")#158329  > 158673

DDoS = pd.read_csv("../flow_labeled/labeld_DDoS.csv")#16050


print("\dataset prepared,cost time:%d" %(time.time() - start))

d0 = data2feature(Benign,0)[:18000]
d1 = data2feature(DoS_GoldenEye,1)
d2 = data2feature(DoS_Hulk,2)
d3 = data2feature(DoS_Slowhttps,3)
d4 = data2feature(DoS_Slowloris,4)
d5 = data2feature(SSH_Patator,5)
d6 = data2feature(FTP_Patator,6)

d7 = data2feature(Web_Attack_BruteForce,7)
d8 = data2feature(Web_Attack_SqlInjection,7)
d9 = data2feature(Web_Attack_XSS,7)

d10 = data2feature(Botnet,8)

d11 = data2feature(PortScan_1,9)
d12 = data2feature(PortScan_2,9)[:15000]

d13 = data2feature(DDoS,10)

Data_tupple = (d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13)

Data = np.concatenate(Data_tupple,axis=0)

Data = discard_fiv_tupple(Data)

np.random.shuffle(Data)

x_raw = np.array(Data[:,:-1],dtype="float32")
x_raw = discard_fiv_tupple(x_raw)
y_raw = np.array(Data[:,-1],dtype="int32")

data_train,data_test,label_train,label_test = train_test_split(x_raw,y_raw,test_size=0.2,random_state=0)
#==========================================================================




#==========================================================================
def labels_transform(mlist,classes):
	
	batch_label = np.zeros((len(mlist),classes),dtype="i4")
	for i in range(len(mlist)):
		batch_label[i][mlist[i]] = 1
	return batch_label
#============================================================================

#parameter
learning_rate = 0.0005
img_shape = 40*40
classes_num = 11
batch_size = tf.placeholder(tf.int32,[])
lstm_input_size = 160
lstm_timestep_size = 10
lstm_hidden_layers = 2
train_iter = 30000

# cnn network

_X = tf.placeholder(tf.float32,[None,img_shape])
y = tf.placeholder(tf.int32,[None,classes_num])
keep_prob = tf.placeholder(tf.float32)

def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="VALID")

def max_pool(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")

#40*40*1
cnn_input = tf.reshape(_X,[-1,40,40,1])

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
conv_1 =  tf.nn.relu(conv2d(cnn_input,W_conv1) + b_conv1)
#36*36*32
pool_1 = max_pool(conv_1)
#18*18*32

W_conv2 = weight_variable([3,3,32,64])
b_conv2 = bias_variable([64])
conv_2 = tf.nn.relu(conv2d(pool_1,W_conv2) + b_conv2)
#16*16*64
pool_2 = max_pool(conv_2)
#8*8*64 = 4096

W_fc1 = weight_variable([8*8*64,1600])
b_fc1 = bias_variable([1600])
pool_2_flat = tf.reshape(pool_2,[-1,8*8*64])
cnn_fc1 = tf.matmul(pool_2_flat,W_fc1) + b_fc1
cnn_fc1_drop = tf.nn.dropout(cnn_fc1,keep_prob)

# LSTM network
lstm_input = tf.reshape(cnn_fc1_drop,[-1,lstm_timestep_size,lstm_input_size])

lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=lstm_hidden_layers,forget_bias=1.0,
	state_is_tuple=True,activation=None)

rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256,256]]

multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

init_state = multi_rnn_cell.zero_state(batch_size,dtype=tf.float32)

outputs,state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,inputs=lstm_input,
	initial_state=init_state,dtype=tf.float32,time_major=False)

h_state = state[-1][1] 

W_lstm = tf.Variable(tf.truncated_normal(shape=[256,classes_num],stddev=0.1),dtype=tf.float32)
bias_lstm = tf.Variable(tf.constant(0.15,dtype=tf.float32,shape=[classes_num]))
logits = tf.matmul(h_state,W_lstm) + bias_lstm



# loss and eval

predictions = {
	"classes":tf.argmax(input=logits,axis=1),
	"probabilities":tf.nn.softmax(logits,name="softmax_tensor")
	}

# loss = -tf.reduce_mean(y*tf.log(predictions["probabilities"]))
loss = tf.losses.mean_squared_error(y,predictions["probabilities"])
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,).minimize(loss)

correct_prediction = tf.equal(predictions["classes"],tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



TP = tf.metrics.true_positives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
FP = tf.metrics.false_positives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
TN = tf.metrics.true_negatives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
FN = tf.metrics.false_negatives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
recall = tf.metrics.recall(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
tf_accuracy = tf.metrics.accuracy(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True

sess = tf.Session()

# begin traing
print("\n"+"="*50 +"Benign Trainging"+"="*50)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())#initializing
_batch_size = 128
mydata_train = DataSet(data_train,label_train)
statr = time.time()
for i in range(train_iter):
	batch = mydata_train.next_batch(_batch_size)
	labels = labels_transform(batch[1],classes_num)
	if (i+1)%200 ==0:

		train_accuracy = sess.run(accuracy,feed_dict={_X:batch[0],y:labels,
			keep_prob:1.0,batch_size:_batch_size})
		
		print("\nthe %dth loop,training accuracy:%f" %(i+1,train_accuracy))
	sess.run(train_op,feed_dict={_X:batch[0],y:labels,keep_prob:0.5,
		batch_size:_batch_size})

print("\ntraining finished cost time:%f" %(time.time() - statr))


#batch test：
test_accuracy = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
test_batch_size = 2000
preLabel = []
mlabel = []
test_iter = len(data_test)//test_batch_size + 1

mydata_test = DataSet(data_test,label_test)
print("\n"+"="*50+"Benign test"+"="*50)
test_start = time.time()
for i in range(test_iter):
	batch = mydata_test.next_batch(test_batch_size)
	mlabel = mlabel + list(batch[1])
	labels = labels_transform(batch[1],classes_num)

	e_accuracy = sess.run(accuracy,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	tensor_tp,value_tp = sess.run(TP,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	tensor_fp,value_fp = sess.run(FP,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	tensor_tn,value_tn = sess.run(TN,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	tensor_fn,value_fn = sess.run(FN,feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size})
	preLabel = preLabel + list(sess.run(predictions["classes"],feed_dict={_X:batch[0],y:labels,keep_prob:1.0,batch_size:test_batch_size}))


	test_accuracy = test_accuracy + e_accuracy
	true_positives = true_positives + value_tp
	false_positives = false_positives + value_fp
	true_negatives = true_negatives + value_tn
	false_negatives = false_negatives + value_fn
	
print("\ntest cost time :%d" %(time.time() - test_start))
print("\n"+"="*50+"Test result"+"="*50)
print("\n test accuracy :%f" %(test_accuracy/test_iter))
print("\n true positives :%d" %true_positives)
print("\n false positives :%d" %false_positives)
print("\n true negatives :%d" %true_negatives)
print("\n false negatives :%d" %false_negatives)
print("\n"+"="*50+"  DataSet Describe  "+"="*50)
print("\nAll DataSet Number:%s Trainging DataSet Number:%s Test DataSet Number:%s" %(len(x_raw),len(data_train),len(data_test)))

mP = true_positives/(true_positives+false_positives)
mR = true_positives/(true_positives+false_negatives)
mF1_score = 2*mP*mR/(mP+mR)

print("\nPrecision:%f" %mP)
print("\nRecall:%f" %mR)
print("\nF1-Score:%f" %mF1_score)
conmat = confusion_matrix(mlabel,preLabel)
print("\nConfusion Matraics:")
print(conmat)
print(len(mlabel))


