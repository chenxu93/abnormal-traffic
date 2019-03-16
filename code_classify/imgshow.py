import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 

def data2feature(f_name,cla):
	file_value = f_name.values
	file_value[:,-1] = cla
	feature = file_value
	feature = feature[:,1:]
	np.random.shuffle(feature)
	return feature

f0 = pd.read_csv("./labeld_Monday-Benign.csv")
f1 = pd.read_csv("./labeld_Wednesday-Dos-GlodenEye-7350.csv")
f2 = pd.read_csv("./labeld_Wednesday-Dos-Hulk5919.csv")
f3 = pd.read_csv("./labeld_Friday-DDoS-16050.csv")
f4 = pd.read_csv("./labeld_Tuesday-FTP-Palator-3945.csv")
f5 = pd.read_csv("./labeld_Wednesday-Dos-slowloris-1916.csv")
f6 = pd.read_csv("./labeld_Wednesday-Dos-slowlhttptest-1403.csv")

type_0 = data2feature(f0,0)[:4]
type_1 = data2feature(f1,1)[:4]
type_2 = data2feature(f2,2)[:4]
type_3 = data2feature(f3,3)[:4]
type_4 = data2feature(f4,4)[:4]
type_5 = data2feature(f5,5)[:4]
type_6 = data2feature(f6,6)[:4]

imgs = np.concatenate((type_0,type_1,type_2,type_3,type_4,type_5,type_6),axis=0)


fig = plt.figure()
for i in range(7):
	for j in range(4):
		ax = fig.add_subplot(7,4,i*4+(j+1))
		cla = (imgs[i*4+j])[:-1]
		img_title = (imgs[i*4+j])[-1]
		img = cla.reshape([40,40])
		ax.imshow(img,cmap='gray')
		plt.axis("off")
		plt.title(img_title)

plt.show()


