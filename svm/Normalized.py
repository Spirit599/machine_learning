import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def krkopt_type(s):
	if len(s) <= 1:
		return ord(s)-ord('a')+1
	else:
		if s == b'draw':
			return 1
		else:
			return -1;

#start
data_path =  os.getcwd()
file_name = '/krkopt.data'
data = np.loadtxt(data_path + file_name, dtype=str, delimiter=',', converters={	0: krkopt_type, 2: krkopt_type, 
																				4: krkopt_type, 6: krkopt_type, })
x, y = np.split(data, (6,), axis=1)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=1, train_size=0.2)
print(x_train.shape)
print(x_test.shape)
x_train_nor = preprocessing.scale(x_train)
x_test_nor = preprocessing.scale(x_test)
print(x_train_nor)
print(x_test_nor.shape)