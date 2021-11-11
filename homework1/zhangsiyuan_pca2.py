import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk

#导入Boston放假数据集
from sklearn.datasets import load_boston
boston=load_boston()

#PCA的手动实现
#这里采用最大方差的观点，设x是m维随机变量，Sigma是x的协方差矩阵，Sigma的特征值从大到小依次是lambda1,lambda2...lambdam>=0
#对应的特征向量依次是alpha1,alpha2...alpham.
#则x的第k主成分是yk=<alphak,x>
#对本例中，x=boston_used
def mypca(data,n=2,m=6,t=10):
    #data为要做pca的随机变量,n为保留的主成分数，m为随机变量维数，t为时间窗长度
    N=int(np.floor(data.shape[0]/t))
    lamb_0=np.zeros([N,m])
    lamb_new_0=np.zeros([N,n])
    for i in range(N):
        #X就是个临时的中转站，不重要，作用就是把每次切片的给搞出来
        #X=np.array([data[i*10:i*10+10,0],data[i*10:i*10+10,1],data[i*10:i*10+10,2],\
        #           data[i*10:i*10+10,3],data[i*10:i*10+10,4],data[i*10:i*10+10,5]]).transpose()
        X=np.array([data[i*t:i*t+t,k] for k in range(m)]).transpose()
        X_mean=np.mean(X, axis=0)#计算原始数据中每一列的均值，即每一个feature的不同观测的均值
        X_new=X-X_mean#将每一个feature的均值都化为0
        X_cov=np.cov(X_new,rowvar=0)#求协方差矩阵，rowvar=0表示每一列是一个feature的不同观测
        lamb,alpha=np.linalg.eig(X_cov) #计算协方差矩阵的特征值和特征向量
        lamb_0[i,:]=lamb#用来存储lamb
        index=np.argsort(lamb) #将特征值按从小到大排序，index保留的是对应alpha中的下标
        index_new=index[-1:-n-1:-1]#取出上面保留的下标中前n个特征值的下标
        #alpha_new=alpha[:, index_new] #取最大的两维特征值对应的特征向量组成变换矩阵
        #X_low=X_new.dot(alpha_new)#求出变换后的主成分，即实现降维
        #写到这里突然发现这道题似乎没有要求算出主成分是什么，只要求出特征值占比就OK了，所以有几行废话我注释掉了...
        lamb_new_0[i,:]=lamb[index_new]
    return lamb_new_0,lamb_0

pc1=np.zeros(50)
pc1_and_pc2=np.zeros(50)
lamb_new_0,lamb_0=mypca(boston.data[0:500,7:13],2)
for i in range(50):
    pc1[i]=lamb_new_0[i,0]/lamb_0[i,:].sum()#第一主成分占比计算
    pc1_and_pc2[i]=(lamb_new_0[i,0]+lamb_new_0[i,1])/lamb_0[i,:].sum()#第二主成分和第一主成分累计占比计算
    
t=np.linspace(1,50,50)
#曲线图（我感觉有点丑）
plt.plot(t,pc1,label=r'$pc_{1}$')
plt.plot(t,pc1_and_pc2,'r',label=r'$pc_{1}+pc_{2}$')
plt.fill_between(t, pc1, pc1_and_pc2, color='orangered')
plt.fill_between(t, pc1, color='dodgerblue')
plt.legend()
plt.show()
#柱状图（哎这个有点好看）
fig, ax = plt.subplots()
bar1=ax.bar(t,pc1_and_pc2, width=0.35, bottom=None,color='r',label=r'$pc_{1}+pc_{2}$')
bar2=ax.bar(t,pc1, width=0.35, bottom=None,color='b',label=r'$pc_{1}$')
ax.set_ylabel('Percent')
plt.legend(bbox_to_anchor=(1.05, 0), loc=1, borderaxespad=0)
plt.show()