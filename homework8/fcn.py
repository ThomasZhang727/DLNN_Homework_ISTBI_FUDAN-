from sklearn.neural_network import MLPClassifier
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

mnist=fetch_openml('mnist_784',version=1)
#对于data和target，都是Dataframe格式，这里将其转化为矩阵格式
X,y=mnist['data'].values,mnist['target'].values
#print(X.shape)
#print(y.shape)
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]#留出法划分数据集

depth=[]
acc=[]
for i in range(9):
    d=i+3
    depth.append(d)
    sizes=[50]*d
    model=MLPClassifier(hidden_layer_sizes=sizes,max_iter=10,alpha=0.01)
    model.fit(X_train,y_train)
    acc.append(model.score(X_test,y_test))
    print('隐藏层数目为',d,'时,全连接网络在测试集上的精确率为:',model.score(X_test,y_test))

plt.plot(depth,acc,'-*')
plt.ylabel('Accuracy on test set')
plt.xlabel('Numbers of hidden layers')
plt.show()