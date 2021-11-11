import numpy as np
import matplotlib.pyplot as plt

#最速下降法
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

# 生成的斐波拉契数列,F(0)=F(1)=1，这是在为使用斐波那契法做一维搜索做准备
def Fab(a,b,delta):
    x=1
    y=1
    l=[1,1]
    while l[-1]<(b-a)/delta:#注意此处是根据Kiefer的证明,得出的Fibonacci方法的收敛速度,即(b-a)/Fn<=delta
        t=y
        y=x+t
        x=t
        l.append(y)
    return l  

#一维搜索，这里我用的是Fibonacci搜索
def Fibonacci_search(a,b,delta,function):
    F=Fab(a,b,delta)
    n=len(F)-1
    k=1
    t1=a+F[n-1]/F[n]*(b-a)#第一次搜索缩小区间
    t2=a+F[n-2]/F[n]*(b-a)#第一次搜索缩小区间
    while k<n-1:
        #通过迭代不断缩小搜索区间
        f1=function(t1)
        f2=function(t2)
        if f1<f2:
            a=t2
            t2=t1
            t1=a+F[n-k-1]/F[n-k]*(b-a)
        else:
            b=t1
            t1=t2
            t2=b-F[n-k-1]/F[n-k]*(b-a)
        k+=1
    return (a+b)/2#最终(a,b)会收敛到一个点(pf:区间套定理)，在有限次迭代后，不妨设这个点为最后一次迭代点的中点


#最速下降法
def fastest_descent(fun,grad,X_0,N,eps=1e-20):
    k=0
    X=X_0
    X_list=[]
    X_list.append(X_0)
    while k<N-2:
        if np.linalg.norm(grad(X))<eps:#对迭代终止条件做一个判断
            break
        else:
            dk=-grad(X)
            def function(t):
                return fun(X+t*dk)
            delta=0.001
            t=Fibonacci_search(0.1,10,delta,function)#应用斐波那契法做一维搜索
            #print(t)
            if X[0]<-1:#这里加一个限制，防止优化结果真的跑到负无穷去
                X[0]=-1
            if X[1]<-1:
                X[1]=-1
            plt.plot(X[0],X[1], 'c*')
            X=X+t*dk
            k+=1
            X_list.append(X)
    return X_list

X_list=fastest_descent(fun,grad,np.array([0.01,0.03],dtype=np.float),1000)
X_array=np.asarray(X_list)
print(X_list)
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