import numpy
import torch

x=torch.rand(3,3)
y=torch.rand(3,3)

print(x+y)

print(torch.add(x,y))

res=torch.empty(3,3)
torch.add(x,y,out=res)
print(res)
y.add_(x)
print(y)

print(x)
print(x[:,-1])
print(x[-1,:])
print(x[1,:]) # 주의. 이건 두 번째 행을 의미한다.

#########################################################################

x=torch.randn(4,4) # 4행 4열
y=x.view(16) # 크기 16의 1차원 벡터.
z=x.view(-1,2) # 8행 2열, -1이 의미하는 바는?

#########################################################################

x=torch.randn(1)
print(x)
print(x.item())

#########################################################################

if torch.cuda.is_available():
    device=torch.device("cuda")
    y=torch.ones_like(x,device=device) # 첫 번째 방법

    x=x.to(device) # 두 번째 방법


