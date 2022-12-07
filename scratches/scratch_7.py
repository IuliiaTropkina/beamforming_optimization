import numpy as np
import numba as nb


test_numpy_dtype = np.dtype([("x", np.int64),("y", np.int64)])
test_numba_dtype = nb.from_dtype(test_numpy_dtype)


@nb.njit
def working_fn(thing):
    thing = np.copy(thing)
    for j in range(len(thing)):
        thing[j]['x'] += j
        thing[j]['y'] += j
    return thing

@nb.njit
def desired_fn(thing):
    thing = np.copy(thing)
    thing['x'] += np.arange(len(thing))
    thing['y'] += np.arange(len(thing))
    return thing


a = np.zeros(3, test_numpy_dtype)
print(a)
print(working_fn(a))
print(desired_fn(a))