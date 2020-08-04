# N rank numpy

import numpy as np

x=np.array([1,2,3])
print(x)

y=np.array([1,2,3],[4,5,6])
print(y)

# zeros, ones, eye

import numpy as np

a=np.zeros((3,2))
print(a)

b=np.ones((1,2))

c=np.eye(2,2)
print(c)

# generation

import numpy as np

x=np.array([1,2])
y=np.array([[1,2],[5,6]])

print(np.add(x,y))
print(np.subtract(x,y))
print(np.multiply)
print(np.matmul(x,y))