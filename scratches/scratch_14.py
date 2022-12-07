import numpy as np
import matplotlib.pyplot as plt
from lib.plots_stuff import axisEqual3D
import scipy.signal


# Node Structure of K-ary Tree
class NewNode():

    def __init__(self, val):
        self.key = val
        # all children are stored in a list
        self.child = {}


N = 64
X = np.zeros([N,N]+[N], dtype=int)
MIP_LEVELS = 3

X[20, 7:25, 0] = 1
X[60, 40:45, 0] = 1
X[3, 0, 1:6] = 1
#X_for_plot = np.array(X,dtype = bool)

indexes = np.where(X)
indexes_reshaped = np.zeros((len(indexes[0]),3))
indexes_reshaped[:,0] = indexes[0]
indexes_reshaped[:,1] = indexes[1]
indexes_reshaped[:,2] = indexes[2]

# plt.figure()
# plt.plot(indexes[0],indexes[1],"o", color = "k")
# plt.show()

#TX = np.array([7, 7])

# def mipmap2(vox, level, node):
#     if level != 0:
#         new_vox = np.ceil(vox / (2 ** (2)))
#         level -= 1
#         #if tuple(new_vox) not in node.child.keys():
#         node = mipmap2(new_vox, level, node)
#         node.child[tuple(new_vox)].child[tuple(vox)] = NewNode(1)
#         return node
#     else:
#         node.child[tuple(vox)] = NewNode(1)
#         return node

def mipmap2(array, level):
    new_array = np.ceil(array / (2 ** (level)))
    node = {}
    for ind_vox, vox in enumerate(array):
        if tuple(new_array[ind_vox]) not in node.keys():
            node[tuple(new_array[ind_vox])] = [ind_vox]
        else:
            node[tuple(new_array[ind_vox])].append(ind_vox)
    return node



# MIP = [indexes_reshaped]
# for l in range(1,MIP_LEVELS+1):
#     root = NewNode(1)
#     MIP.append(mipmap2(indexes_reshaped, l, root))
stage_level = 1
MIP_arrays = [indexes_reshaped]
MIP_dicts = []
for l in range(1,MIP_LEVELS+1):
    MIP_dicts.append(mipmap2(MIP_arrays[l-1], stage_level))
    MIP_arrays.append(np.array(list(MIP_dicts[l-1].keys())))

#aaaaaa = np.array(list(a.keys()))

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


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(MIP_arrays[l][:,0],MIP_arrays[l][:,1],MIP_arrays[l][:,2], marker="o", color = "k")
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




