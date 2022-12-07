import numpy as np
import matplotlib.pyplot as plt
from lib.plots_stuff import axisEqual3D
import scipy.signal

N = 64
X = np.zeros([N,N]+[N], dtype=int)
MIP_LEVELS = 3

X[20, 7:25, 0] = 1
X[60, 40:45, 0] = 1
X[3, 0, 1:6] = 1
#X_for_plot = np.array(X,dtype = bool)

# indexes = np.where(X_for_plot)
# plt.figure()
# plt.plot(indexes[0],indexes[1],"o", color = "k")
# plt.show()

#TX = np.array([7, 7])

def mipmap(array):
    M_size = 2
    M = np.ones([M_size, M_size]+[M_size])
    new_array_size = int(len(array)/M_size)
    new_array = np.zeros((new_array_size,new_array_size,new_array_size), dtype = int)
    for x in range(0,new_array_size):
        for y in range(0,new_array_size):
            for z in range(0, new_array_size):
                new_array[x,y,z]=sum(sum(sum(array[x*M_size:x*M_size+M_size, y*M_size:y*M_size+M_size, z*M_size:z*M_size+M_size]*M)))

    return new_array


MIP = [X]
for l in range(MIP_LEVELS):
    print(l)
    MIP.append(mipmap(MIP[l]))

# def mip_coord(x, lvl):
#     return x // (2**lvl)


for l in range(MIP_LEVELS+1):
    # IMG = np.zeros(list(MIP[l].shape) + [3])
    #
    # IMG[:, :, 2]= MIP[l]
    # x,y = mip_coord(TX, l)
    # IMG[x, y, 0] = 1
    # #R = scipy.signal.convolve2d(MIP[l], np.ones([3, 3]), mode='same')
    # #IMG[:, :, 1] = R
    # IMG[-1,-1,1] = 1
    indexes = np.where(MIP[l])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(indexes[0],indexes[1],indexes[2], marker="o", color = "k")
    ax.set_xticks(np.linspace(0, 64, 5, dtype=int)// (2**l))
    ax.set_yticks(np.linspace(0, 64, 5, dtype=int) // (2**l))
    # if  len(vertices)!=0:
    #ax.plot(X, Y, zs=Z, marker=".", linestyle=None)
    axisEqual3D(ax)



    #ax.plot(indexes[0],indexes[1],indexes[2],"o", color = "k")


    #ax.imshow(MIP[l])
    # ax.set_xticks(np.linspace(0, 64, 5, dtype=int)// (2**l))
    # ax.set_yticks(np.linspace(0, 64, 5, dtype=int) // (2**l))
    #ax.set_title(f"MIP level {l}")
plt.show()




