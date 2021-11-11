import numpy as np
import matplotlib.pyplot as plt

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

def SGD(fun,grad,X_0,N,eps=1e-20,lr=0.01):
    #fun,grad为优化函数和优化函数的梯度函数，X_0为初值，N为最大迭代次数，eps为终止条件，默认值为10^(-20)，lr为学习率，默认为0，01
    k=0
    X=X_0
    X_list=[]
    X_list.append(X_0)
    while k<N-2:
        if np.linalg.norm(grad(X))<eps:#对迭代终止条件做一个判断
            break
        else:
            dk=-grad(X)
            #print(t)
            if X[0]<-1:#这里加一个限制，防止优化结果真的跑到负无穷去
                X[0]=-1
            if X[1]<-1:
                X[1]=-1
            plt.plot(X[0],X[1], 'c*')
            X=X+lr*dk
            k+=1
            X_list.append(X)
    return X_list

X_list=SGD(fun,grad,np.array([0.01,0.03],dtype=np.float),1000,eps=1e-20,lr=0.01)
X_array=np.asarray(X_list)
#print(X_list)
x=np.linspace(-1,1,1000)
y=np.linspace(-1,1,1000)
xx,yy=np.meshgrid(x,y)
zz=fun([xx,yy])
plt1=plt.contourf(xx,yy,zz,cmap=plt.cm.hot)
#画等高线
plt.contour(xx,yy,zz)
#设置颜色条，（显示在图片右边）
plt.colorbar(plt1)
plt.plot(X_array[:,0],X_array[:,1],'b')
plt.show()

X_list_1=SGD(fun,grad,np.array([-0.01,-0.03],dtype=np.float),1000,eps=1e-20,lr=0.01)
X_array_1=np.asarray(X_list_1)
#print(X_list)
x=np.linspace(-1,1,1000)
y=np.linspace(-1,1,1000)
xx,yy=np.meshgrid(x,y)
zz=fun([xx,yy])
plt2=plt.contourf(xx,yy,zz,cmap=plt.cm.hot)
#画等高线
plt.contour(xx,yy,zz)
#设置颜色条，（显示在图片右边）
plt.colorbar(plt2)
plt.plot(X_array_1[:,0],X_array_1[:,1],'b')
plt.show()