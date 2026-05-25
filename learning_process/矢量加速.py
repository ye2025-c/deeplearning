import torch
from Timer import Timer
from d2l import torch as d2l

'''比较矢量加速和纯Python循环的速度'''

n=1000
a=torch.ones([n])
b=torch.ones([n])

c=torch.zeros([n])
time=Timer()

for i in range(n):
    c[i]=a[i]+b[i]

print(f'{time.stop():.5f} sec')

time.start()
d=a+b

print(f'{time.stop():.5f} sec')