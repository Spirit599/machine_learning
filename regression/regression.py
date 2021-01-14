#import sys
import pandas as pd
import numpy as np
#from google.colab import drive
#!gdown --id '1wNKAxQ29G15kgpBy_asjTcZRRgmsCZRm' --output data.zip
#!unzip data.zip
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



mouth_num = 12
day_num = 20
hour_num = 24
sample_feature_num = 18
hours_of_period = 8
training_num_per_month = day_num * hour_num - hours_of_period #471
final_training_length = training_num_per_month * mouth_num #5652
final_feature = sample_feature_num * hours_of_period #162
dim = final_feature + 1  #dimension维度  因为常数项的存在，所以dimension需要多加一栏
test_num = 240



#load 'train.csv'
data = pd.read_csv('./train.csv', encoding = 'big5')

#preprocessing
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()


#extract features1
month_data = {}
for month in range(mouth_num):
	sample = np.empty([sample_feature_num, day_num * hour_num]) #十八项测试数据，20天每天24小时，480小时
	for day in range(day_num):
		sample[:, day * hour_num : 
		(day + 1) * hour_num] = raw_data[sample_feature_num * (day_num * month + day) : 
										sample_feature_num * (day_num * month + day + 1), :]
	month_data[month] = sample

#extract features2
x = np.empty([final_training_length, final_feature], dtype = float)
y = np.empty([final_training_length, 1], dtype = float)
for month in range(mouth_num):
	for day in range(day_num):
		for hour in range(hour_num):
			if day == 19 and hour > hour_num - hours_of_period - 1:
				continue
			#把一个月的所有行(18行)，每隔9列数据放成一行
			x[month * training_num_per_month + day * hour_num + hour, :] = month_data[month][:,day * hour_num + hour : day * hour_num + hour + hours_of_period].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
			#pm2.5那行数据
			y[month * training_num_per_month + day * hour_num + hour, 0] = month_data[month][9, day * hour_num + hour + hours_of_period] #value
#print(month_data[0][:,0:9])# 1月1号0-8点18项数据，有18行
#print(x[0,:])# 1月1号0-8点18项数据,全部组成一行，只有一个list
#print(month_data[0][9,9]) # 1月1号9点pm2.5的数据



'''
#Normalize 手写
mean_x = np.mean(x, axis = 0) #每列的平均值，就是把每天的同一个时间点的同一种数据做平均值
std_x = np.std(x, axis = 0)  #每列的标准差，就是把每天的同一个时间点的同一种数据做标准差
for i in range(len(x)):
  for j in range(len(x[0])):
    if std_x[j] != 0:
      x[i][j] = (x[i][j] - mean_x[j]) / std_x[j]  #x的每一个数，减去当前列的平均数（mean函数），得到的差值除以当前列的标准差
                                #就是将同一种类型的数据调整范围区间，做归一化
x
'''

#preprocessing Normalize 
x = preprocessing.scale(x)
#x_train_nor = preprocessing.scale(x_train)
#x_test_nor = preprocessing.scale(x_test)



'''
#Split Training Data Into "train_set" and "validation_set"
import math
x_train_set = x[: math.floor(len(x)*0.8), :]  #x的前80%用于训练集
y_train_set = y[: math.floor(len(y)*0.8), :]  #y的前80%用于训练集
x_validation = x[math.floor(len(x) *0.8): , :]  #x的后20%用于验证集
y_validation = y[math.floor(len(y) *0.8): , :]  #y的后20%用于验证集
'''

#Split Training Data Into "train_set" and "validation_set"
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=1, train_size=0.8)

w = np.zeros([dim, 1])
x = np.concatenate((np.ones([final_training_length, 1]), x), axis = 1).astype(float) #等式右边的x是归一化后的数据,np.ones([12*471, 1])为了与x的数据
learning_rate = 100
iter_time = 10000   #学习次数
adagrad = np.zeros([dim, 1])
eps = 0.0000000001  #为了避免adagrad的分母为0而加的极小数值
for t in range(iter_time):
	loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/final_training_length) #rmse #loss的平方和再除以471除以12（471*12是总数据量）再开根号
	if(t % 100 == 0): #每训练100次打印一次loss值
		print(str(t) + "次training loss为:" + str(loss))
	gradient = 2 * np.dot(x.transpose(), np.dot(x, w) - y) #dim*1
	adagrad += gradient ** 2
	w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
print(gradient)
np.save('weight.npy', w)
w


# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('./test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:-1]  #除去文件中第一列的‘id_'和名称，仅留数据
testdata[test_data == 'NR'] = 0  #将"NR"的数据改为0
test_data = test_data.to_numpy()
test_x = np.empty([test_num, final_feature], dtype = float) #18*9是feature的数据量, test_num组数据，每组18*9个feature
for i in range(test_num):
	test_x[i, :] = test_data[sample_feature_num * i : sample_feature_num * (i + 1), :].reshape(1, -1) #每隔18个数据为一行
'''
for i in range(len(test_x)):     #行
	for j in range(len(test_x[0])): #列
		if std_x[j] != 0:
			test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
'''
test_x = preprocessing.scale(test_x)


test_x = np.concatenate((np.ones([test_num, 1]), test_x), axis = 1).astype(float)
test_x



w = np.load('weight.npy')
ans_y = np.dot(test_x, w)

testdata = testdata.iloc[:,-1] #期待的pm25
#testdata[test_data == 'NR'] = 0  #将"NR"的数据改为0
testdata = testdata.to_numpy()
e_pm25_y = np.empty([test_num, 1], dtype = float)
for i in range(test_num):
	e_pm25_y[i] = testdata[9 + i * sample_feature_num]


total_loss = e_pm25_y - ans_y
np.set_printoptions(suppress=True)
total_loss = np.power(total_loss, 2)
mean_loss = np.sqrt(np.sum(total_loss) / test_num)
print(mean_loss)
'''
import csv
with open('submit.csv', mode='w', newline='') as submit_file:
	csv_writer = csv.writer(submit_file)
	header = ['id', 'value']
	print(header)
	csv_writer.writerow(header)
	for i in range(test_num):
		row = ['id_' + str(i), ans_y[i][0]]
		csv_writer.writerow(row)
		print(row)
'''