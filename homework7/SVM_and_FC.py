import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import datetime

mnist=fetch_openml('mnist_784',version=1)
#对于data和target，都是Dataframe格式，这里将其转化为矩阵格式
X,y=mnist['data'].values,mnist['target'].values
#print(X.shape)
#print(y.shape)
#图片的可视化，注意此时图片数据是以向量格式存储的，需要对其重塑为28*28的矩阵
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.axis('off')
    test=X[i,:].reshape(28,28)
    plt.imshow(test)
    plt.title(y[i])
plt.show()
#统计不同标签对应的图片数据的个数
nums=np.zeros([10,1])
n=np.arange(0,10)
for i,num in enumerate(y):
    k=int(num)
    nums[k,0]+=1
#绘制不同标签对应图片数据个数的直方图
rect=plt.bar(n,height=nums.reshape(-1),width=0.4,alpha=0.8,color='blue')
for r in rect:
    height=int(r.get_height())
    plt.text(r.get_x()+r.get_width()/2,height+1,str(height),ha="center",va="bottom")
plt.show()
#训练
X_train,X_test,y_train,y_test=X[:60000],X[60000:],y[:60000],y[60000:]#留出法划分数据集
starttime = datetime.datetime.now()#计时开始，计时对象：训练
model=SVC(kernel='rbf')
model.fit(X_train,y_train)
endtime = datetime.datetime.now()#计时结束
print('训练时间为',(endtime - starttime).seconds,'s')
#测试
starttime = datetime.datetime.now()#计时开始，计时对象：测试
print('在测试集上的精确率为',model.score(X_test,y_test))
endtime = datetime.datetime.now()#计时结束
print('测试时间为',(endtime - starttime).seconds,'s')

cm=confusion_matrix(y_test,model.predict(X_test),labels=model.classes_)#计算混淆矩阵
disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)

disp.plot()#绘制混淆矩阵
plt.show()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import datetime

transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
#！！！请注意如果是macOS，需要更改root为非根目录，亲测会报错
trainsets=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainsets,batch_size=100,shuffle=True)
testsets=torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testsets,batch_size=100,shuffle=False)

class fcn(nn.Module):
    def __init__(self):
        super(fcn,self).__init__()
        self.linear1=nn.Linear(784,512)
        self.linear2=nn.Linear(512,256)
        self.linear3=nn.Linear(256,128)
        self.linear4=nn.Linear(128,64)
        self.linear5=nn.Linear(64,10)

    def forward(self,x):
        x=F.relu(self.linear1(x))
        #print(x.shape)
        x=F.relu(self.linear2(x))
        #print(x.shape)
        x=F.relu(self.linear3(x))
        #print(x.shape)
        x=F.relu(self.linear4(x))
        #print(x.shape)
        x=self.linear5(x)
        #print(x.shape)
        return x

net=fcn()
if torch.cuda.is_available():
    if torch.cuda.device_count()>1:
        #注意初始化只能初始化一次，如需多次运行请注释掉代码
        torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)#初始化
        net=net.cuda()
        net=nn.parallel.DistributedDataParallel(net)
    else:
        net=net.cuda()
loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
starttime = datetime.datetime.now()#计时开始，计时对象：训练
for epoch in range(20):
    for x,y in trainloader:
        if torch.cuda.is_available():
            x=x.cuda()
            y=y.cuda()
        x=x.view(x.size(0),-1)
        y_hat=net(x)
        l=loss(y_hat,y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l_=l.mean().item()
    print('epoch:',epoch+1,',loss:',l_)
endtime = datetime.datetime.now()#计时结束
print('全连接网络训练时间为:',(endtime - starttime).seconds,'s')

net.eval()
acc=0
sum=0
for x,y in testloader:
    if torch.cuda.is_available():
            x=x.cuda()
            y=y.cuda()
    x=x.view(x.size(0),-1)
    y_hat=net(x)
    predict=torch.argmax(y_hat,dim=1)
    for j in range(predict.size()[0]):
        if int(predict[j])==int(y[j]):
            acc+=1
        sum+=1
acc=acc/sum
print('全连接网络的的精确率为:',acc) 