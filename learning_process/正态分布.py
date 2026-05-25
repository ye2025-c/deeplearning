import math
from d2l import torch as d2l
import torch
import numpy as np


def normal(x,mu,sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(-0.5 / sigma**2 * (x - mu)**2)

#可视化
x=np.arange(-7.5,7.5,0.1)
params = [(0,1),(0,2),(0,3),(1,1),(2,1),(3,1)]

d2l.plot(x,[normal(x,mu,sigma) for mu,sigma in params],xlabel='x',ylabel='p(x)',legend=[f'mu={mu},sigma={sigma}' for mu,sigma in params])
d2l.plt.show()