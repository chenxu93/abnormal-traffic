import xgboost as xgb 
import pandas as pd 
import numpy as np 
import time
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



def data2feature(f_name,cla):
	file_value = f_name.values
	file_value[:,-1] = cla
	feature = file_value
	feature = feature[:,1:]
	np.random.shuffle(feature)
	return feature


def discard_fiv_tupple(data):
	
	for i in range(10):
		data[7+i*160] = 0
		data[10+i*160:22+i*160] = 0
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


for x in range(10):
	Data[:,10+160*x:21+160*x] = 0

x_raw = np.array(Data[:,:-1],dtype="float32")
y_raw = np.array(Data[:,-1],dtype="int32")

data_train,data_test,label_train,label_test = train_test_split(x_raw,y_raw,test_size=0.25,random_state=0)


dtrain = xgb.DMatrix(data_train,label=label_train)
dtest = xgb.DMatrix(data_test,label=label_test)

param = {}

param["objective"] = "multi:softmax"
param["eta"] = 0.2
param["gama"] = 1
param["max_depth"] = 8
param["silent"] = 1
param["num_class"] = 2
# param["eval_metric"] = auc

num_round = 10
watchlist = [(dtrain,"train"),(dtest,"test")]

bst = xgb.train(param,dtrain,num_round,watchlist)

pred = bst.predict(dtest)

print("\nModel report")

print("\nAccuracy:%f" %metrics.accuracy_score(label_test,pred))
print("\nPrecision:%f" %metrics.average_precision_score(label_test,pred))
print("\nRecall:%f" %metrics.recall_score(label_test,pred))
print("\nF1-score:%f" %metrics.f1_score(label_test,pred))
print("\nconfusion matrix:" )
print("\n%s" %metrics.confusion_matrix(label_test,pred))
# pred("\nF1 score:%f" %metrics.f1_score(label_test,pred))



feature_importance = bst.get_score(fmap="",importance_type="weight")
print("\nfeature importance：")
print(feature_importance)
mfile = open("feature_importance_2.txt","w")

for k,v in feature_importance.items():
	mfile.write(str(k)+":"+str(v)+"\n")
mfile.close()


fig,ax = plt.subplots(figsize=(100,50))
xgb.plot_importance(bst,ax=ax)
plt.show()
print("done!")
