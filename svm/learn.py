import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn import model_selection
 
# Import some data to play with
iris = datasets.load_iris()
X = iris.data#得到样本集
y = iris.target#得到标签集
 
##变为2分类
X, y = X[y != 2], y[y != 2]#通过取y不等于2来取两种类别
 
# Add noisy features to make the problem harder添加扰动
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]
 

print(X.shape)
print(y.shape)

# shuffle and split training and test sets划分样本集
train_data, test_data, train_label, test_label = model_selection.train_test_split(X, y, test_size=.3,random_state=0)
#train_data用于训练的样本集, test_data用于测试的样本集, train_label训练样本对应的标签集, test_label测试样本对应的标签集
 
# Learn to predict each class against the other分类器设置
svm = svm.SVC(kernel='linear', probability=True,random_state=random_state)#使用核函数为线性核，参数默认，创建分类器
 


###通过decision_function()计算得到的test_predict_label的值，用在roc_curve()函数中
test_predict_label = svm.fit(train_data, train_label).decision_function(test_data)
#首先通过fit来对训练样本和训练样本标签进行训练得到模型，然后通过decision_function来获得模型对于测试样本集预测的标签集



 
# Compute ROC curve and ROC area for each class#计算tp,fp
#通过测试样本输入的标签集和模型预测的标签集进行比对，得到fp,tp,不同的fp,tp是算法通过一定的规则改变阈值获得的
fpr,tpr,threshold = roc_curve(test_label, test_predict_label) ###计算真正率和假正率
#print(fpr)
#print(tpr)
#print(threshold)
roc_auc = auc(fpr,tpr) ###计算auc的值，auc就是曲线包围的面积，越大越好
print(roc_auc)

plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
