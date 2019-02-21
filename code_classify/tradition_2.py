import pandas as pd 
from sklearn import metrics
import os
from sklearn.model_selection import train_test_split
import numpy as np 
import time 

def knn_classifier(feature,label):
	 from sklearn.neighbors import KNeighborsClassifier
	 model = KNeighborsClassifier()
	 model.fit(feature,label)
	 return model

def logistic_regression_classifier(feature,label):
	from sklearn.linear_model import LogisticRegression  
	model = LogisticRegression(penalty='l2')  
	model.fit(feature, label)  
	return model  

def random_forest_classifier(feature, label):  
    from sklearn.ensemble import RandomForestClassifier  
    model = RandomForestClassifier(n_estimators=8)  
    model.fit(feature, label)  
    return model

def decision_tree_classifier(feature, label):  
    from sklearn import tree  
    model = tree.DecisionTreeClassifier()  
    model.fit(feature, label)  
    return model  

def gradient_boosting_classifier(feature, label):  
    from sklearn.ensemble import GradientBoostingClassifier  
    model = GradientBoostingClassifier(n_estimators=200)  
    model.fit(feature, label)  
    return model 

def svm_classifier(feature, label):  
    from sklearn.svm import SVC  
    model = SVC(kernel='rbf', probability=True)  
    model.fit(feature, label)  
    return model 


def data2feature(f_name,cla):
	file_value = f_name.values
	file_value[:,-1] = cla
	feature = file_value
	feature = feature[:,1:]
	np.random.shuffle(feature)
	return feature


if __name__ == '__main__':  

	test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
	classifiers = {'KNN':knn_classifier,  
	               'LR':logistic_regression_classifier,  
	               'RF':random_forest_classifier,  
	               'DT':decision_tree_classifier,  
	              #'SVM':svm_classifier,  
	             'GBDT':gradient_boosting_classifier  
	}  


	print("\ndata preparing ... ... ... ")

	#读取数据

	def data_prepare(f1_name,f2_name,y1,y2):
		d1 = f1_name.values
		d2 = f2_name.values
		d1[:,-1] = y1
		d2[:,-1] = y2

		dataset = np.concatenate((d1,d2),axis=0)

		#打乱
		np.random.shuffle(dataset)
		return dataset


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

	#由于Heartbleed和Infiltraton攻击非常少，在做多分类的时候，并不考虑这两类攻击
	#多分类 做11分类 正常+10类攻击

	#二分类可考虑Heartbleed和Infiltraton攻击

	print("\n数据加载完成，耗时：%d" %(time.time() - start))

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

	x_raw = np.array(Data[:,1:-1],dtype="float32")
	y_raw = np.array(Data[:,-1],dtype="int32")

	data_train,data_test,label_train,label_test = train_test_split(x_raw,y_raw,test_size=0.25,random_state=0)

	from sklearn import tree
	clf = tree.DecisionTreeClassifier()
	clf.fit(data_train,label_train)
	# pred = clf.predict(data_test)
	print(clf.score(data_test,label_test))



	of = open('report_2.txt','w')
	for classifier in test_classifiers:
        
		print('******************* %s ********************' % classifier)  
		of.write('******************* %s ********************\n' % classifier)  
		start_time = time.time()  
		model = classifiers[classifier](data_train, label_train)  
		print('training took %fs!' % (time.time() - start_time))
		of.write('training took %fs!\n' % (time.time() - start_time))  
		predict = model.predict(data_test)  
		of.write ('classify_report\n')
		classify_report = metrics.classification_report(label_test, predict)   #使用这种模式
		print("\nAccuracy:%f" %metrics.accuracy_score(label_test,predict))
		print("\nPrecision:%f" %metrics.average_precision_score(label_test,predict))
		print("\nRecall:%f" %metrics.recall_score(label_test,predict))
		print("\nF1-score:%f" %metrics.f1_score(label_test,predict))
		print("\nconfusion matrix:")
		print("\n%s" %metrics.confusion_matrix(label_test,predict))
		print(classify_report)
		of.write(classify_report)
     
	of.close()