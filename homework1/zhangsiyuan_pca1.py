#导入Boston放假数据集
from sklearn.datasets import load_boston
boston=load_boston()
boston_used=boston.data[0:500,7:13]
labels=boston['feature_names'][7:13]

#调用sklearn.decomposition.PCA实现PCA
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

pc1=np.zeros(50)
pc1_and_pc2=np.zeros(50)
for i in range(50):
    pca = PCA(n_components=2)
    #X就是个临时的中转站，不重要，作用就是把每次切片的给搞出来
    X=np.array([boston_used[i*10:i*10+10,0],boston_used[i*10:i*10+10,1],boston_used[i*10:i*10+10,2],\
       boston_used[i*10:i*10+10,3],boston_used[i*10:i*10+10,4],boston_used[i*10:i*10+10,5]]).transpose()
    pca.fit(X)
    pc1[i]=pca.explained_variance_ratio_[0]
    pc1_and_pc2[i]=pca.explained_variance_ratio_.sum(0)
    
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


