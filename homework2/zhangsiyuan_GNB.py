import numpy as np
import sklearn
from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

x1,y1=make_moons(n_samples=100, noise=0.3, random_state=0)#生成月牙形数据，noise代表了噪音，random_state代表了噪声生成的随机数种子
x2,y2=make_circles(n_samples=100, noise=0.2, random_state=0, factor=0.5)#生成环形数据，noise代表了噪音，random_state代表了噪声生成的随机数种子，factor代表了内圈与外圈的比
x3,y3=make_classification(n_samples=100,n_features=2,n_classes=2,n_clusters_per_class=1,n_redundant=0,n_informative=2)#n_features代表，特征个数= n_informative + n_redundant + n_repeated（默认均为0），n_calsses代表了类数，n_clusters_per_class代表了每类的聚类数        
data_names=['make_moons','make_circles','make_classification']
data=[(x1,y1),(x2,y2),(x3,y3)]#将三个数据集集合在一起以便于循环程序

plt.figure(figsize=(5,9))

for i,d in enumerate(data):
    x,y=d#读入训练集
    model=GaussianNB()#简历模型,模型为Gausssian朴素贝叶斯模型
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,shuffle=True)#将数据集按照3：1的比例划分为训练集和测试集并打乱
    model.fit(x_train,y_train)#对模型进行训练
    print(data_names[i],'数据的模型准确率为:',model.score(x_test,y_test))#对模型准确率进行评估
    x_min, x_max = x[:, 0].min()-0.5,x[:,0].max()+0.5
    y_min, y_max = x[:, 1].min()-0.5,x[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
    xy=np.c_[xx.ravel(),yy.ravel()]
    y_pred=model.predict(xy).reshape(xx.shape)
    color_map=plt.cm.RdBu#将颜色选定为红蓝，与分类对应的画图颜色统一
    zz=model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(xx.shape)#画出决策边界，.ravel即拉平数据集
    plt.subplot(3,1,i+1)
    plt.title(data_names[i])
    plt.contourf(xx, yy, zz, alpha=0.3, cmap=color_map)
    p1=plt.scatter(x[y==0,0], x[y==0, 1], color='red')#画出第一类点
    p2=plt.scatter(x[y==1,0], x[y==1, 1], color='blue')#画出第二类点
    plt.text(xx.max()-0.5,yy.max()-0.5,model.score(x_test,y_test))

plt.show()                             