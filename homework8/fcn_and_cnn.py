import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import datetime
import numpy as np
import matplotlib.pyplot as plt

transform=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize([0.5],[0.5])])
#！！！请注意如果是macOS，需要更改root为非根目录，亲测会报错
trainsets=torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainsets,batch_size=256,shuffle=True)
testsets=torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform)
testloader=torch.utils.data.DataLoader(testsets,batch_size=256,shuffle=False)

def count_parameters(model):
    l=[]
    for p in model.parameters():
        l.append(p.numel())
    return np.array(l).sum()

class fcn3(nn.Module):
    def __init__(self,fc1_output):
        super(fcn3,self).__init__()
        self.linear1=nn.Linear(1024,fc1_output)
        self.linear2=nn.Linear(fc1_output,fc1_output//2)
        self.linear3=nn.Linear(fc1_output//2,fc1_output//4)
        self.linear4=nn.Linear(fc1_output//4,fc1_output//8)
        self.linear5=nn.Linear(fc1_output//8,10)

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

class cnn3(nn.Module):
    def __init__(self,fc1_output=64):
        super(cnn3,self).__init__()
        self.conv1=nn.Conv2d(1,16,3)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv2=nn.Conv2d(16,64,3)
        self.pool2=nn.MaxPool2d(2,2)
        self.conv3=nn.Conv2d(64,128,3)
        self.pool3=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(128*2*2,fc1_output)
        self.fc2=nn.Linear(fc1_output,10)
        
    def forward(self,x):
        x=self.pool1(torch.relu(self.conv1(x)))
        x=self.pool2(torch.relu(self.conv2(x)))
        x=self.pool3(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=self.fc2(x)
        return x

class cnn11(nn.Module):
    def __init__(self,fc1_output=64):
        super(cnn11,self).__init__()
        self.conv1=nn.Conv2d(1,4,3)
        self.conv2=nn.Conv2d(4,16,3)
        self.pool1=nn.MaxPool2d(2,2)
        self.conv3=nn.Conv2d(16,64,3)
        self.pool2=nn.MaxPool2d(2,2)
        self.conv4=nn.Conv2d(64,256,3)
        self.pool3=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(256*2*2,fc1_output)
        self.fc2=nn.Linear(fc1_output,fc1_output//2)
        self.fc3=nn.Linear(fc1_output//2,fc1_output//4)
        self.fc4=nn.Linear(fc1_output//4,10)
        
    def forward(self,x):
        x=torch.relu(self.conv1(x))
        x=torch.relu(self.conv2(x))
        x=self.pool1(x)
        x=torch.relu(self.conv3(x))
        x=self.pool2(x)
        x=torch.relu(self.conv4(x))
        x=self.pool3(x)
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.relu(self.fc3(x))
        x=self.fc4(x)
        return x


param_cnn3=[]
acc_cnn3=[]
fc1_outputs=[64,128,256]
for fc1_output in fc1_outputs:
    net=cnn3(fc1_output=fc1_output)
    param_cnn3.append(count_parameters(net))
    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            nn.DataParallel(net)
        net=net.cuda()
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
    starttime = datetime.datetime.now()#计时开始，计时对象：训练
    for epoch in range(5):
        for index,(x,y) in enumerate(trainloader):
            if torch.cuda.is_available():
                x=x.cuda()
                y=y.cuda()
            y_hat=net(x)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        l_=l.mean().item()
        print('epoch:',epoch+1,',loss:',l_)
    endtime = datetime.datetime.now()#计时结束
    print('CNN训练时间为:',(endtime - starttime).seconds,'s')

    net.eval()
    acc=0
    sum=0
    for x,y in testloader:
        if torch.cuda.is_available():
            x=x.cuda()
            y=y.cuda()
        y_hat=net(x)
        predict=torch.argmax(y_hat,dim=1)
        for j in range(predict.size()[0]):
            if int(predict[j])==int(y[j]):
                acc+=1
            sum+=1
    acc=acc/sum
    acc_cnn3.append(acc)
    print('CNN的的精确率为:',acc) 

param_fcn3=[]
acc_fcn3=[]
fc1_outputs=[128,256,512]
for fc1_output in fc1_outputs:
    net=fcn3(fc1_output=fc1_output)
    param_fcn3.append(count_parameters(net))
    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            nn.DataParallel(net)
        net=net.cuda()
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
    starttime = datetime.datetime.now()#计时开始，计时对象：训练
    for epoch in range(5):
        for index,(x,y) in enumerate(trainloader):
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
    print('FCN训练时间为:',(endtime - starttime).seconds,'s')

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
    acc_fcn3.append(acc)
    print('FCN的的精确率为:',acc) 

param_cnn11=[]
acc_cnn11=[]
fc1_outputs=[64,128,256]
for fc1_output in fc1_outputs:
    net=cnn11(fc1_output=fc1_output)
    param_cnn11.append(count_parameters(net))
    if torch.cuda.is_available():
        if torch.cuda.device_count()>1:
            nn.DataParallel(net)
        net=net.cuda()
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=0.01)
    starttime = datetime.datetime.now()#计时开始，计时对象：训练
    for epoch in range(5):
        for index,(x,y) in enumerate(trainloader):
            if torch.cuda.is_available():
                x=x.cuda()
                y=y.cuda()
            y_hat=net(x)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        l_=l.mean().item()
        print('epoch:',epoch+1,',loss:',l_)
    endtime = datetime.datetime.now()#计时结束
    print('CNN训练时间为:',(endtime - starttime).seconds,'s')

    net.eval()
    acc=0
    sum=0
    for x,y in testloader:
        if torch.cuda.is_available():
            x=x.cuda()
            y=y.cuda()
        y_hat=net(x)
        predict=torch.argmax(y_hat,dim=1)
        for j in range(predict.size()[0]):
            if int(predict[j])==int(y[j]):
                acc+=1
            sum+=1
    acc=acc/sum
    acc_cnn11.append(acc)
    print('CNN的的精确率为:',acc) 

plt.plot(param_cnn11,acc_cnn11,'g-*',label='CNN_11')
plt.plot(param_cnn3,acc_cnn3,'r-*',label='CNN_3')
plt.plot(param_fcn3,acc_fcn3,'b-*',label='FCN_3')
plt.legend()
plt.show()