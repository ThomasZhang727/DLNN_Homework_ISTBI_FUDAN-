import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import sklearn.metrics as sm
import sklearn
from sklearn.cluster import KMeans,MiniBatchKMeans
from timeit import timeit

iris=datasets.load_iris()
X=iris['data'][:,(2,3)]#这里选取叶子的长度和宽度作为数据
y=iris['target'].astype(np.float64)

plt.scatter(X[:,0],X[:,1],marker='o',c=y)#绘制可视化图片
plt.show()  

k=3#选择聚类为3类
kmeans=KMeans(n_clusters=k)
kmeans.fit(X,y)#模型训练
pred_y=kmeans.predict(X)#为计算轮廓系数做准备

#绘制Voronoi图
mins=X.min(axis=0) - 0.1
maxs=X.max(axis=0) + 0.1
xx, yy=np.meshgrid(np.linspace(mins[0], maxs[0], 1000),np.linspace(mins[1], maxs[1], 1000))
Z=kmeans.predict(np.vstack([xx.ravel(), yy.ravel()]).T)
Z=Z.reshape(xx.shape)

plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),cmap="Pastel2")
plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),linewidths=1, colors='k')
plt.scatter(X[:,0],X[:,1],marker='o',c=y)
plt.show()

print('轮廓系数为:',sm.silhouette_score(X, pred_y, sample_size=len(X), metric='euclidean'))#计算轮廓系数

#比较算法minibatchkmeans与kmeans的效率，发现前者更高，用时更少
'''
times=np.empty((100, 2))
for k in range(1, 101):
    kmeans=KMeans(n_clusters=k, random_state=42)
    minibatch_kmeans=MiniBatchKMeans(n_clusters=k, random_state=42)
    print('\r{}/{}'.format(k, 100), end='')
    times[k-1, 0]=timeit('kmeans.fit(X)', number=10, globals=globals())
    times[k-1, 1]=timeit('minibatch_kmeans.fit(X)', number=10, globals=globals())

plt.plot(range(1, 101), times[:, 0], 'r--', label='K-Means')
plt.plot(range(1, 101), times[:, 1], 'b.-', label='Mini-batch K-Means')
plt.xlabel('$k$', fontsize=16)
plt.title('Training Time (seconds)',fontsize=14)
plt.axis([1, 100, 0, 6])
plt.show()
'''

#计算不同k下的不同轮廓系数并作图
num_clusters=np.arange(8)+2
score=[]
for i in num_clusters:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(X,y)
    pred_y=kmeans.predict(X)
    score.append(sm.silhouette_score(X, pred_y, sample_size=len(X), metric='euclidean'))
    
plt.plot(num_clusters,score)

plt.scatter(X[:,0],X[:,1],'o',c=kmeans.labels_)
plt.show()