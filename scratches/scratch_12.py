import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

# def plot_map(coordinates, MIP_LEVELS):
#     f, axes = plt.subplots(1, MIP_LEVELS + 1)
#     for ax, l in zip(axes, range(MIP_LEVELS + 1)):
#         IMG = np.zeros(list(MIP[l].shape) + [3])
#
#         IMG[:, :, 2] = MIP[l]
#         x, y = mip_coord(TX, l)
#         IMG[x, y, 0] = 1
#         # R = scipy.signal.convolve2d(MIP[l], np.ones([3, 3]), mode='same')
#         # IMG[:, :, 1] = R
#         IMG[-1, -1, 1] = 1
#         ax.imshow(IMG)
#         ax.set_xticks(np.linspace(0, 64, 5, dtype=int) // (2 ** l))
#         ax.set_yticks(np.linspace(0, 64, 5, dtype=int) // (2 ** l))
#         ax.set_title(f"MIP level {l}")
N = 64
X = np.zeros([N,N], dtype=int)
MIP_LEVELS = 3

X[20, 7:25] = 1
X[60, 40:45] = 1

#X_for_plot = np.array(X,dtype = bool)

# indexes = np.where(X_for_plot)
# plt.figure()
# plt.plot(indexes[0],indexes[1],"o", color = "k")
# plt.show()

TX = np.array([7, 7])

def mipmap(x):
    R = scipy.signal.convolve2d(x, np.ones([3, 3]), mode='same')
    return R[::2, ::2]


MIP = [X]
for l in range(MIP_LEVELS):
    MIP.append(mipmap(MIP[l]))

def mip_coord(x, lvl):
    return x // (2**lvl)

f, axes = plt.subplots(1, MIP_LEVELS + 1)
for ax, l in zip(axes, range(MIP_LEVELS+1)):
    IMG = np.zeros(list(MIP[l].shape) + [3])

    IMG[:, :, 2]= MIP[l]
    x,y = mip_coord(TX, l)
    IMG[x, y, 0] = 1
    #R = scipy.signal.convolve2d(MIP[l], np.ones([3, 3]), mode='same')
    #IMG[:, :, 1] = R
    IMG[-1,-1,1] = 1
    ax.imshow(IMG)
    ax.set_xticks(np.linspace(0, 64, 5, dtype=int)// (2**l))
    ax.set_yticks(np.linspace(0, 64, 5, dtype=int) // (2**l))
    ax.set_title(f"MIP level {l}")
plt.show()




