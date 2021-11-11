import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt

x_train=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
y_train=np.array([-1,1,1,-1])

def kernel(X1,X2):#自定义核函数，具体为（1+X1^TX2)^2
    return (1+X1.dot(X2.T))**2

model=SVC(kernel=kernel)
model.fit(x_train,y_train)#训练model，具体模型为核SVM，所使用的核函数为上面定义核函数

x_lim,y_lim=[-5,5],[-5,5]#定义画图范围
x=np.linspace(x_lim[0],x_lim[1],500)
y=np.linspace(y_lim[0],y_lim[1],500)
xx,yy=np.meshgrid(x,y)
xy=np.vstack([xx.ravel(),yy.ravel()]).T
d=model.decision_function(xy).reshape(xx.shape)#画出分类曲线
Z=model.predict(xy).reshape(xx.shape)#画出不同类所在的区域
color_map=plt.cm.RdBu
plt.contourf(xx,yy,Z,cmap=color_map,alpha=0.3)
plt.contour(xx,yy,d,cmap=color_map)
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap='RdBu')
plt.show()