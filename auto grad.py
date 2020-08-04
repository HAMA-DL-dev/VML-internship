import torch
x=torch.ones(2,2,requires_grad=True)
print(x)

y=x+2
print(y)
print(y.grad_fn)
print(y.requires_grad) # x에 의한 연산으로 계산되는 y. 이때 x의 re~은 true임

z=3*y*y
out=z.mean()
print(z)
print(out)

#########################################################################

a=torch.randn(2,2)
a=((a*3)/(a-1))
print(a.requires_grad) # False
a.requires_grad_(True)
print(a.requires_grad) # True

b=(a*a).sum()
print(b.grad_fn)
print(b.requires_grad)

out.backward()
print(x.grad)

#########################################################################
x=torch.randn(3,requires_grad=True)
print(x)
y=x*2
cnt=1

while y.data.norm()<1000:
    y=y*2
    cnt=cnt+1

print(y)
print(cnt)
v=torch.tensor([0.1,1.0,1e-4],dtype=torch.float)
y.backward(v)
print(x.grad)