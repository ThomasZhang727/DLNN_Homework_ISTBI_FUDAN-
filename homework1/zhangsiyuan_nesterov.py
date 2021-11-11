import numpy as np
import matplotlib.pyplot as plt


def fun(X):
    x=X[0]
    y=X[1]
    return x**3+y**3-3*x*y

def grad(X):
    #定义梯度，计算可得gradf(x,y)=[3x**2-3y,3y**2-3x]
    x=X[0]
    y=X[1]
    return np.array([3*x**2-3*y,3*y**2-3*x])

def nesterov(fun,grad,X0,beta=0.1,rhot=0.1,N=1000):
    mt=-grad(X0)
    t=0
    Xt=X0
    while t<N-1:
        mt=beta*mt-rhot*grad(Xt+beta*mt)
        plt.plot(Xt[0],Xt[1],'c*')#可视化过程
        Xt+=mt
        print(Xt)
        t+=1

nesterov(fun,grad,X0=np.array([0.01,0.03],dtype=np.float),beta=0.1,rhot=0.1,N=1000)
#X_array=np.asarray(X)
x=np.linspace(-1.5,1.5,1000)
y=np.linspace(-1.5,1.5,1000)
xx,yy=np.meshgrid(x,y)
zz=fun([xx,yy])
plt1=plt.contourf(xx,yy,zz,cmap=plt.cm.hot)
#画等高线
plt.contour(xx,yy,zz)
#设置颜色条，（显示在图片右边）
plt.colorbar(plt1)
#plt.plot(X_array[:,0],X_array[:,1],'b')
plt.show()