import torch
x=torch.empty(5,3)
print(x)

x=torch.rand(7,5)
print(x)

x=torch.zeros(9,2,dtype=torch.long)
print(x)
x.type()

import numpy as np
x=torch.tensor([10,5,3.1])
y=torch.tensor(np.random.rand(5,3))
print(x)
print(y)

np_v=x.numpy()
print(type(np_v))
print(np_v.dtype)
print(np_v.shape)