import numpy as np
import matplotlib.pyplot as plt

#梯度下降法 gradient_descent
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

def adam(fun,grad,rho1,rho2,theta0,N,delta=1e-8,lr=0.001):
    #fun为目标函数，grad为目标函数的梯度
    #lr为步长，默认值为0.001，rho1,rho2为指数衰减速率，其取值必须在[0,1)区间内
    #delta为用于数值稳定的小常数，默认值为1e-8  
    #theta0为优化函数自变量的初始值
    #N为最大迭代步长
    theta=theta0
    s=np.zeros_like(theta0)
    r=np.zeros_like(theta0)
    t=0
    X_list=[]
    X_list.append(theta0)
    while t<N-1:
        g=grad(theta)#计算梯度
        t+=1
        s=rho1*s+(1-rho1)*g#更新有偏一阶估计
        r=rho2*r+(1-rho2)*(g**2)#更新有偏二阶估计
        s_hat=s/(1-rho1**t)#修正一阶矩的偏差
        r_hat=r/(1-rho2**t)#修正二阶矩的偏差
        delta_theta=-1*lr*s_hat/(np.sqrt(r_hat)+delta)#计算更新量
        plt.plot(theta[0],theta[1],'c*')#可视化过程
        theta+=delta_theta#更新
        X_list.append(theta)
        #print(theta)
    return X_list

X=adam(fun,grad,rho1=0.9,rho2=0.999,theta0=np.array([0.01,0.03],dtype=np.float),N=2000,delta=1e-8,lr=0.1)
X_array=np.asarray(X)
x=np.linspace(-1.5,1.5,1000)
y=np.linspace(-1.5,1.5,1000)
xx,yy=np.meshgrid(x,y)
zz=fun([xx,yy])
plt1=plt.contourf(xx,yy,zz,cmap=plt.cm.hot)
#画等高线
plt.contour(xx,yy,zz)
#设置颜色条，（显示在图片右边）
plt.colorbar(plt1)
plt.plot(X_array[:,0],X_array[:,1],'b')
plt.show()
