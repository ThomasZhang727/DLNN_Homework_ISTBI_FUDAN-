import numpy as np
import matplotlib.pyplot as plt

#牛顿法
def fun(X):
    #定义要求解的函数f(x,y)=x**3+y**3-3xy
    x=X[0]
    y=X[1]
    return x**3+y**3-3*x*y

def grad(X):
    #定义梯度，计算可得gradf(x,y)=[3x**2-3y,3y**2-3x]
    x=X[0]
    y=X[1]
    return np.array([3*x**2-3*y,3*y**2-3*x])

def Hessian(X):
    #定义Hessian矩阵，计算可得H=[[6x,-3],[-3,6y]]
    x=X[0]
    y=X[1]
    return np.array([[6*x,-3],[-3,6*y]])

def newton_descent(fun,grad,Hessian,X_0,N,eps=0.0001):
    k=0
    X=X_0
    X_list=[]
    X_list.append(X_0)
    while k<N-2:
        if np.linalg.norm(grad(X))<eps:
            break
        else:
            pk=-np.linalg.inv(Hessian(X)).dot(grad(X))#更新方向的计算
            plt.plot(X[0],X[1], 'b*')
            X=X+pk#应用更新
            if X[0]<-5:#这里做一个限制，防止跑到负无穷去
                X[0]=-5
            if X[1]<-5:
                X[1]=-5
            k+=1
            X_list.append(X)
    return X_list

X_list=newton_descent(fun,grad,Hessian,np.array([1,2]),N=1000,eps=0.0001)
X_array=np.asarray(X_list)
x=np.linspace(-2,2,1000)
y=np.linspace(-2,2,1000)
xx,yy=np.meshgrid(x,y)
zz=fun([xx,yy])
plt1=plt.contourf(xx,yy,zz,cmap=plt.cm.hot)
#画等高线
plt.contour(xx,yy,zz)
#设置颜色条，（显示在图片右边）
plt.colorbar(plt1)
plt.plot(X_array[:,0],X_array[:,1],'b')
plt.show()