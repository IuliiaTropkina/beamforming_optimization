import numpy as np
from time import time

from lib.numba_opt import njit

N = 2**28
data = np.zeros(N, dtype=np.uint8)

Q = 10000000
#indices = np.random.randint(0, N, Q)
indices = np.arange(0, Q)



@njit
def bla(dat, idx):
    s = 0
    for i in idx:
        s+=dat[i]
    return s

_ = bla(data, np.zeros(5, dtype=int))

t1 = time()
_ = bla(data, indices)


print(time() - t1)