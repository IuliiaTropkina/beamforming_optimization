from lib.numba_opt import njit
from lib.vectors import vector_normalize, norm, vector, origin
import requests
import numpy as np
from physics_experiments.HT_2d_ray_tracer import ray_tracer_2d
import matplotlib.pyplot as plt
from lib.plots_stuff import axisEqual3D
from scipy.spatial.transform import Rotation as R
import trimesh
from math import floor, ceil
from lib.vectors import vector_normalize, norm, vector, origin
from matplotlib.patches import Rectangle


def plot_square(square_center, sz, ax, **kwargs):
    ax.plot([square_center[0] - sz, square_center[0] - sz], [square_center[1] + sz, square_center[1] - sz], **kwargs)
    ax.plot([square_center[0] - sz, square_center[0] + sz], [square_center[1] - sz, square_center[1] - sz], **kwargs)
    ax.plot([square_center[0] + sz, square_center[0] + sz], [square_center[1] + sz, square_center[1] - sz], **kwargs)
    ax.plot([square_center[0] + sz, square_center[0] - sz], [square_center[1] + sz, square_center[1] + sz], **kwargs)


def voxelize_box(box, mip = 0):
    unit_sz = 2**mip
    for x in np.arange(floor(box[0][0]/unit_sz), ceil(box[0][1]/unit_sz)):
        for y in np.arange(floor(box[1][0]/unit_sz), ceil(box[1][1]/unit_sz)):
             yield x, y




def clip_box(box1, box2):
    res = np.zeros([2,2], dtype=float)
    for i in range(2):
        dim1 = box1[i]
        dim2 = box2[i]
        res[i] = np.clip(dim2, dim1[0], dim1[1])
    return res


#print(clip_box([[3,4], [1,2]], [[3.5,7], [1.5, 6]]))



C = np.array([0.3,.12,.6])
SZ = 5
FULL_LOOKUP_BOX = [[C[0] - SZ/2,C[0] + SZ/2], [C[1] - SZ/2, C[1] + SZ/2]]
print(FULL_LOOKUP_BOX)
fig, ax = plt.subplots()
plt.title("MIP 2")
plot_square(C, SZ/2,ax)
M=2
TOLOOK = []
for x,y in voxelize_box(FULL_LOOKUP_BOX, mip=M):
    s = 2 ** M

    TOLOOK.append([x, y])
    ax.add_patch(Rectangle((x*s, y*s), s, s,
                                               edgecolor='pink',
                                               facecolor='blue',
                                               fill=True,
                                               alpha=0.1,
                                               lw=5))
plt.axis("equal")
print(TOLOOK)

fig, ax = plt.subplots()
plt.title("MIP 1")
plot_square(C, SZ/2,ax)
M=1
TOLOOK2 = []
for x,y in TOLOOK:
    s0 = 2 ** M+1
    s = 2 ** M
    box = clip_box([[x*s0, x*s0+s0], [y*s0, y*s0+s0]], FULL_LOOKUP_BOX)
    # print(f"Looking into {(x,y)}, box {box}")
    for xi, yi in  voxelize_box(box, mip=M):

        TOLOOK2.append([xi,yi])
        ax.add_patch(Rectangle((xi*s, yi*s), s, s,
                                                   edgecolor='black',
                                                   facecolor='red',
                                                   fill=True,
                                                   alpha=0.1,
                                                   lw=5))
    break


plt.axis("equal")



print(TOLOOK2)

fig, ax = plt.subplots()
plt.title("MIP 0")
plot_square(C, SZ/2,ax)
M=0
TOLOOK3 = []
for x,y in TOLOOK2:
    s0 = 2 ** M+1
    s = 2 ** M
    box = clip_box([[x*s0, x*s0+s0], [y*s0, y*s0+s0]], FULL_LOOKUP_BOX)
    print(f"Looking into {(x,y)}, box {box}")
    for xi, yi in  voxelize_box(box, mip=M):
        TOLOOK3.append([xi,yi])
        ax.add_patch(Rectangle((xi*s, yi*s), s, s,
                                                   edgecolor='black',
                                                   facecolor='red',
                                                   fill=True,
                                                   alpha=0.1,
                                                   lw=5))



plt.show()