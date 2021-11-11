from sklearn.datasets import load_boston
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score,mean_squared_error
import time
#导入数据集
boston=load_boston()
X,y=boston.data[:,0:13],boston.target
#将数据集按照3：1的比例划分为训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,shuffle=True,random_state=42)
#对数据集进行标准化处理
from sklearn.preprocessing import StandardScaler
ss_X=StandardScaler()   
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1, 1))  
y_test=ss_y.transform(y_test.reshape(-1, 1))
#引入模型，模型为SGDRegressor，即使用SGD方法进行线性回归，默认损失函数为平方损失函数
start=time.time()
model=SGDRegressor()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)
end=time.time()
print('模型使用SGD作为优化器的得分为',model.score(X_test,y_test))
print('模型使用SGD作为优化器的r2_score为',r2_score(y_test,y_predict))
print('模型使用SGD作为优化器的MSE为',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))
print('使用SGD作为优化器的训练与测试用时为:',end-start,'s')

start1=time.time()
model1=SGDRegressor()
model1.fit(X_train,y_train)
y_predict1=model1.predict(X_test)
end1=time.time()
print('线性回归模型的得分为',model1.score(X_test,y_test))
print('线性回归模型的r2_score为',r2_score(y_test,y_predict1))
print('线性回归模型的MSE为',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict1)))
print('线性回归训练与测试用时为:',end1-start1,'s')