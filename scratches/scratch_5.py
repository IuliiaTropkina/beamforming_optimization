import numba

@numba.njit
def aaa():
    x = {1:(1,2,3), 5:(5,6,4)}
    return x

@numba.njit
def b():
    x = aaa()
    x[2]=(1,7,6)
    return x[5]

print(b())