import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F # ReLU 함수 포함

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,5) # 6개의 커널
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1=nn.Linear(16*5*5,120) # layer 1
        self.fc2 = nn.Linear(120,84)   # layer 2
        self.fc3 = nn.Linear(84,10)    # layer 3


    def forward(self,x):
            x=F.max_pool2d(F.relu(self.conv1(x)),(2,2))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
            x=x.view(-1,self.num_flat_features(x)) # 벡터 데이터로 만듬
            x = F.relu(self.fct1(x))
            x = F.relu(self.fct2(x))
            x = self.fct3(x)
            return x

    def num_flat_features(self,x):
            size=x.size()[1:]
            num_features=1
            for s in size:
                num_features *= s
            return num_features

net=Net()
print(net)

#######################################################################333

params=list(net.parameters())
print(len(params))
print(params[0].size())
print(params[1].size())
print(params[1])

input=Variable(torch.randn(1,1,32,32))
out=net(input)
out2=net.forward(input)
print(out)
print(out2)

##########################################################################

net.zero_grad()
out.backward(torch.randn(1,10))

import torch.nn as nn
output=net(input)
target=Variable(torch.arange(1,11,dtype=torch.float))
target=target.view(1,-1)
print(target)
print(output)
criterion=nn.MSELoss()

loss=criterion(output,target)
print(loss)

########################################################3
learning_rate=0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)