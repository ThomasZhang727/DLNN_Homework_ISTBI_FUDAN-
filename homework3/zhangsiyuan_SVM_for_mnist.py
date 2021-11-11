import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import datetime

mnist=fetch_openml('mnist_784',version=1)
#对于data和target，都是Dataframe格式，这里将其转化为矩阵格式
X,y=mnist['data'].values,mnist['target'].values
#print(X.shape)
#print(y.shape)
#图片的可视化，注意此时图片数据是以向量格式存储的，需要对其重塑为28*28的矩阵
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    test=X[i,:].reshape(28,28)
    plt.imshow(test)
    plt.title(y[i])
plt.show()
#统计不同标签对应的图片数据的个数
nums=np.zeros([10,1])
n=np.arange(0,10)
for i,num in enumerate(y):
    k=int(num)
    nums[k,0]+=1
#绘制不同标签对应图片数据个数的直方图
rect=plt.bar(n,height=nums.reshape(-1),width=0.4,alpha=0.8,color='blue')
for r in rect:
    height=int(r.get_height())
    plt.text(r.get_x()+r.get_width()/2,height+1,str(height),ha="center",va="bottom")
plt.show()
#训练
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]#留出法划分数据集
starttime = datetime.datetime.now()#计时开始，计时对象：训练
model=SVC(kernel='rbf')
model.fit(X_train,y_train)
endtime = datetime.datetime.now()#计时结束
print('训练时间为',(endtime - starttime).seconds,'s')
#测试
starttime = datetime.datetime.now()#计时开始，计时对象：测试
print('在测试集上的精确率为',model.score(X_test,y_test))
endtime = datetime.datetime.now()#计时结束
print('测试时间为',(endtime - starttime).seconds,'s')

cm=confusion_matrix(y_test,model.predict(X_test),labels=model.classes_)#计算混淆矩阵
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)

disp.plot()#绘制混淆矩阵
plt.show()
