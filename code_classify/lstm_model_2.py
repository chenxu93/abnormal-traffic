#-*- coding:utf-8 -*-
import tensorflow as tf 
import numpy as np 
from inpudata import DataSet
import time
import os
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.metrics import confusion_matrix




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

d0 = data2feature(Benign,0)
d1 = data2feature(DoS_GoldenEye,1)
d2 = data2feature(DoS_Hulk,1)
d3 = data2feature(DoS_Slowhttps,1)
d4 = data2feature(DoS_Slowloris,1)
d5 = data2feature(SSH_Patator,1)
d6 = data2feature(FTP_Patator,1)

d7 = data2feature(Web_Attack_BruteForce,1)
d8 = data2feature(Web_Attack_SqlInjection,1)
d9 = data2feature(Web_Attack_XSS,1)

d10 = data2feature(Botnet,1)

d11 = data2feature(PortScan_1,1)
d12 = data2feature(PortScan_2,1)

d13 = data2feature(DDoS,1)


Data_tupple = (d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13)


Data = np.concatenate(Data_tupple,axis=0)

Data = discard_fiv_tupple(Data)

np.random.shuffle(Data)

x_raw = np.array(Data[:,:-1],dtype="float32")
# x_raw = discard_fiv_tupple(x_raw)

y_raw = np.array(Data[:,-1],dtype="int32")

data_train,data_test,label_train,label_test = train_test_split(x_raw,y_raw,test_size=0.2,random_state=0)
#==========================================================================




#==========================================================================
def labels_transform(mlist,classes):
	
	batch_label = np.zeros((len(mlist),classes),dtype="i4")
	for i in range(len(mlist)):
		batch_label[i][mlist[i]] = 1
	return batch_label


lr = 0.0001

batch_size = tf.placeholder(tf.int32,shape=[])

input_size = 160

timestep_size = 10

hidden_size = 256

layer_num = 2

class_num = 2

_X = tf.placeholder(tf.float32,[None,timestep_size*input_size])
y = tf.placeholder(tf.int32,[None,class_num])
keep_prob = tf.placeholder(tf.float32)



X = tf.reshape(_X,[-1,timestep_size,input_size])


lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size,forget_bias=1.0,
	state_is_tuple=True,activation=None)


rnn_layers = [tf.nn.rnn_cell.LSTMCell(size) for size in [256,256]]


multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

init_state = multi_rnn_cell.zero_state(batch_size,dtype=tf.float32)


outputs,state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,inputs=X,
	initial_state=init_state,dtype=tf.float32,time_major=False)

h_state = state[-1][1]


W = tf.Variable(tf.truncated_normal(shape=[hidden_size,class_num],stddev=0.1),dtype=tf.float32)
bias = tf.Variable(tf.constant(0.15,dtype=tf.float32,shape=[class_num]))
#[batch_size,hidden_size]*[hidden_size,class_num] + [class_num] --> [batch_size,class_num]
logits = tf.matmul(h_state,W) + bias



predictions = {
	"classes":tf.argmax(input=logits,axis=1),
	"probabilities":tf.nn.softmax(logits,name="softmax_tensor")
	}

# loss = -tf.reduce_mean(y*tf.log(predictions["probabilities"]))
loss = tf.losses.mean_squared_error(y,predictions["probabilities"])
train_op = tf.train.AdamOptimizer(learning_rate=lr,).minimize(loss)

correct_prediction = tf.equal(predictions["classes"],tf.argmax(y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


TP = tf.metrics.true_positives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
FP = tf.metrics.false_positives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
TN = tf.metrics.true_negatives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
FN = tf.metrics.false_negatives(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
recall = tf.metrics.recall(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])
tf_accuracy = tf.metrics.accuracy(labels=tf.argmax(y,axis=1),predictions=predictions["classes"])

print("\n"+"="*50 +"Benign Trainging"+"="*50)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
_batch_size = 128
mydata_train = DataSet(data_train,label_train)
statr = time.time()
for i in range(2000):
	batch = mydata_train.next_batch(_batch_size)
	labels = labels_transform(batch[1],class_num)
	if (i+1)%200 ==0:

		train_accuracy = sess.run(accuracy,feed_dict={_X:batch[0],y:labels,
			keep_prob:1.0,batch_size:_batch_size})
		
		print("\nthe %dth loop,training accuracy:%f" %(i+1,train_accuracy))
	sess.run(train_op,feed_dict={_X:batch[0],y:labels,keep_prob:0.5,
		batch_size:_batch_size})

print("\ntraining finished cost time:%f" %(time.time() - statr))





test_accuracy = 0
true_positives = 0
false_positives = 0
true_negatives = 0
false_negatives = 0
preLabel = []
mlabel = []
test_batch_size = 2000
test_iter = len(data_test)//test_batch_size + 1

mydata_test = DataSet(data_test,label_test)
print("\n"+"="*50+"Benign test"+"="*50)
test_start = time.time()
for i in range(test_iter):
	batch = mydata_test.next_batch(test_batch_size)
	mlabel = mlabel + list(batch[1])
	labels = labels_transform(batch[1],class_num)

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
