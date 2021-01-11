#multiclass classification
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
import matplotlib.pyplot as plt

def krkopt_type(s):
	if len(s) <= 1:
		return ord(s)-ord('a') + 1
	else:
		if s == b'draw':
			return 1
		else:
			return 0;

def err(fpr,tpr):
	length = len(fpr)
	min_dis = 1
	cur_dis = 1
	fin_fpr = 0
	fin_tpr = 0
	for i in range(length):
		cur_dis = abs(fpr[i] + tpr[i] - 1)
		if(cur_dis < min_dis):
			min_dis = cur_dis
			fin_tpr = tpr[i]
			fin_fpr = fpr[i]
	return fin_fpr,fin_tpr


#start
data_path =  os.getcwd()
file_name = '/krkopt.data'
data = np.loadtxt(data_path + file_name, dtype=str, delimiter=',', converters={	0: krkopt_type, 2: krkopt_type, 
																				4: krkopt_type, 6: krkopt_type, })
x, y = np.split(data, (6,), axis = 1)
#x = np.transpose(x)
#y = np.transpose(y)
x = x.astype(np.float64)
y = y.astype(np.float64)
y = y.flatten()#扁平化

(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=1, train_size=0.2)


#标准归一化
#x_train_nor = preprocessing.scale(x_train)
#x_test_nor = preprocessing.scale(x_test)
#x_train = x_train_nor
#x_test = x_test_nor

clf = svm.SVC(C=16, kernel='rbf', gamma=0.125, decision_function_shape='ovr')
y_predict = clf.fit(x_train, y_train).decision_function(x_test)


fpr,tpr,threshold = roc_curve(y_test, y_predict) ###计算真正率和假正率

roc_auc = auc(fpr,tpr) ###计算auc的值，auc就是曲线包围的面积，越大越好
roc_err_fpr,roc_err_tpr = err(fpr,tpr)

print(roc_err_fpr,roc_err_tpr)

print(clf.score(x_train, y_train))
y_hat = clf.predict(x_train)
print(clf.score(x_test, y_test))
y_hat = clf.predict(x_test)
print(clf.intercept_)#b值


lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([1, 0], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()