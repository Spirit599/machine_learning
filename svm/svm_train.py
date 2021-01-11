#multiclass classification
import numpy as np
import os
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


fold_num = 5
max_accuracy = 0
final_accuracy = 0
curr_C = 2**(-5) #(2^(-5),2^15)
max_C = 2**10
min_C = 2**(-5)
best_C = 0
curr_gamma = 2**(-15) #(2^(-15),2^3)
max_gamma = 2**1
best_gamma = 0


def krkopt_type(s):
	if len(s) <= 1:
		return ord(s)-ord('a')+1
	else:
		if s == b'draw':
			return 1
		else:
			return -1;

def divide_data(fold_num):
	temp_x_train = x_train
	temp_y_train = y_train
	names = globals()
	for i in range(1, fold_num + 1):
		names['x_train_%s' % i] = i
		names['y_train_%s' % i] = i	
	for i in range(1, fold_num + 1):
		if i <= fold_num - 1:
			(names['x_train_%s' % i],temp_x_train,
			names['y_train_%s' % i],temp_y_train) = train_test_split(temp_x_train, temp_y_train, 
																	random_state = 1, 
																	train_size = 1 / (fold_num + 1 - i))
		else:
			names['x_train_%s' % i] = temp_x_train
			names['y_train_%s' % i] = temp_y_train	




def fold_check(fold_num):
	names = globals()
	accumulate_accuracy = 0
	curr_accuracy = 0
	global curr_C #(2^(-5),2^15)
	global curr_gamma #(2^(-15),2^3)
	global max_accuracy
	global best_gamma
	global best_C
	while(curr_gamma <= max_gamma):
		curr_gamma = curr_gamma *2
		curr_C = min_C
		while(curr_C <= max_C):
			curr_C = curr_C * 2
			accumulate_accuracy = 0
			for i in range(1, fold_num + 1):
				diff_num = 0
				for j in range(1, fold_num + 1):
					if i != j:
						diff_num = diff_num + 1
						if diff_num == 1:
							curr_x_train = names['x_train_%s' % j]
							curr_y_train = names['y_train_%s' % j]
						else:
							curr_x_train = np.append(curr_x_train, names['x_train_%s' % j], axis = 0)
							curr_y_train = np.append(curr_y_train, names['y_train_%s' % j], axis = 0)		

				curr_x_test = names['x_train_%s' % i]
				curr_y_test = names['y_train_%s' % i]
				#print("----第{0}次交叉检验----".format(i))
				#print(curr_x_train.shape)
				#print(curr_y_train.shape)
				#print(curr_x_test.shape)
				#print(curr_y_test.shape)		

				clf = svm.SVC(C=curr_C, kernel='rbf', gamma=curr_gamma, decision_function_shape='ovr')
				clf.fit(curr_x_train, curr_y_train.ravel())
				curr_accuracy = clf.score(curr_x_test, curr_y_test)
				#print(curr_accuracy)
				accumulate_accuracy = accumulate_accuracy + curr_accuracy
			#print("----平均准确率----")
			curr_accuracy = accumulate_accuracy/fold_num
			#print(curr_accuracy)
			print(curr_accuracy)
			if(curr_accuracy > max_accuracy):
				max_accuracy = curr_accuracy
				best_gamma = curr_gamma
				best_C = curr_C
			print('curr_C='+str(curr_C),'curr_gamma='+str(curr_gamma),'curr_accuracy='+str(curr_accuracy))
			print('best_C='+str(best_C),'best_gamma='+str(best_gamma),'max_accuracy='+str(max_accuracy))
	


#start
data_path =  os.getcwd()
file_name = '/krkopt.data'
data = np.loadtxt(data_path + file_name, dtype=str, delimiter=',', converters={	0: krkopt_type, 2: krkopt_type, 
																				4: krkopt_type, 6: krkopt_type, })
x, y = np.split(data, (6,), axis=1)
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=1, train_size=0.2)

#标准归一化
#x_train_nor = preprocessing.scale(x_train)
#x_test_nor = preprocessing.scale(x_test)
#x_train = x_train_nor
#x_test = x_test_nor

print(x_train.shape)
divide_data(fold_num)
print("start---fold__check")
fold_check(fold_num)
print(best_C,best_gamma,max_accuracy)#得到C=16,gamma=0.125
"""
clf = svm.SVC(C=16, kernel='rbf', gamma=0.125, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

print(clf.score(x_train, y_train))
y_hat = clf.predict(x_train)

print(clf.score(x_test, y_test))
y_hat = clf.predict(x_test)

"""