import torch
import numpy as np
import matplotlib.pyplot as plt

x=torch.tensor([0.01,0.03],dtype=float)#设置初始值
x.requires_grad=True#引入梯度

optimizer=torch.optim.SGD([x],lr=0.01)#引入SGD优化器

def fun(X):
    #定义要求解的函数f(x,y)=x**3+y**3-3xy
    x=X[0]
    y=X[1]
    return x**3+y**3-3*x*y

for epoch in range(20000):
    p=fun(x)#目标函数  
    optimizer.zero_grad()#初始化优化器  
    p.backward()  #反向传播
    optimizer.step() #优化器迭代
    plt.plot(x.detach().numpy()[0],x.detach().numpy()[1], 'c*')#可视化求解过程

x=np.linspace(-1,1,1000)
y=np.linspace(-1,1,1000)
xx,yy=np.meshgrid(x,y)
zz=fun([xx,yy])
plt1=plt.contourf(xx,yy,zz,cmap=plt.cm.hot)
#画等高线
plt.contour(xx,yy,zz)
#设置颜色条，（显示在图片右边）
plt.colorbar(plt1)
plt.show()