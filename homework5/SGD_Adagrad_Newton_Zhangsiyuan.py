import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.autograd import Variable
import torch

boston=load_boston()
X,y=boston.data[:,0:13],boston.target.reshape(-1,1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,shuffle=True,random_state=42)

ss_X=StandardScaler()   
ss_y=StandardScaler()
X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)
y_train=ss_y.fit_transform(y_train.reshape(-1, 1))  
y_test=ss_y.transform(y_test.reshape(-1, 1))

def model(x,w,b):
    return x@w.t()+b

loss=torch.nn.MSELoss()

X_train=torch.Tensor(X_train)
y_train=torch.Tensor(y_train)
X_test=torch.Tensor(X_test)
y_test=torch.Tensor(y_test)
w0=torch.randn(13,requires_grad=True)#模型初始化
b0=torch.randn(1,requires_grad=True)#模型初始化

w,b=w0,b0#为了更好的比较，伪随机初始化
for epoch in range(2000):
    l=loss(model(X_train,w,b),y_train) 
    grad_w=torch.autograd.grad(l,w,create_graph=True)
    Hessian=torch.tensor([])
    for anygrad in grad_w[0]:  # torch.autograd.grad返回的是元组
        Hessian=torch.cat((Hessian, torch.autograd.grad(anygrad,w,retain_graph=True)[0]))
    Hessian=Hessian.view(w.size()[0], -1)
    Hinv=torch.linalg.inv(Hessian)
    w=Variable(w-grad_w[0]@Hinv,requires_grad=True)
    grad_b=torch.autograd.grad(l,b,create_graph=True)
    Hessian=torch.tensor([])
    for anygrad in grad_b[0]:  # torch.autograd.grad返回的是元组
        Hessian=torch.cat((Hessian, torch.autograd.grad(anygrad,b,retain_graph=True)[0]))
    Hessian=Hessian.view(b.size()[0], -1)
    Hinv=torch.linalg.inv(Hessian)
    b=Variable(b-grad_b[0]@Hinv,requires_grad=True)

MSE0=loss(model(X_test,w,b),y_test).item()
print('线性回归模型利用Newton法优化后在测试集上的MSE为:',MSE0)

w,b=w0,b0#为了更好的比较，伪随机初始化
optimizer1=torch.optim.SGD([w,b],lr=0.01)

for epoch in range(2000):
    l=loss(model(X_train,w,b),y_train) 
    optimizer1.zero_grad()  
    l.backward()  
    optimizer1.step() 

MSE1=loss(model(X_test,w,b),y_test).item()
print('线性回归模型利用SGD优化后在测试集上的MSE为:',MSE1)

w,b=w0,b0#为了更好的比较，伪随机初始化
optimizer2=torch.optim.Adagrad([w,b],lr=0.01)

for epoch in range(2000):
    l=loss(model(X_train,w,b),y_train) 
    optimizer2.zero_grad()  
    l.backward()  
    optimizer2.step() 

MSE2=loss(model(X_test,w,b),y_test).item()
print('线性回归模型利用Adagrad优化后在测试集上的MSE为:',MSE2)

