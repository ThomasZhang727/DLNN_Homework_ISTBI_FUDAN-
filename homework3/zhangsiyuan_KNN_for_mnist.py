import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import datetime

mnist=fetch_openml('mnist_784',version=1)
#对于data和target，都是Dataframe格式，这里将其转化为矩阵格式
X,y=mnist['data'].values,mnist['target'].values
#print(X.shape)
#print(y.shape)
#训练
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]#留出法划分数据集
starttime = datetime.datetime.now()#计时开始，计时对象：训练
knn_clf=KNeighborsClassifier(n_neighbors=4,weights='distance')
knn_clf.fit(X_train,y_train)
endtime = datetime.datetime.now()#计时结束
print('训练时间为',(endtime - starttime),'s')
#测试
starttime = datetime.datetime.now()#计时开始，计时对象：测试
print('模型在测试集的准确率为:',knn_clf.score(X_test,y_test))
endtime = datetime.datetime.now()#计时结束
print('测试时间为',(endtime - starttime).seconds,'s')
