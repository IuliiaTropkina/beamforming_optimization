from lib.vectors import vector_normalize, norm, vector, origin
import requests
import numpy as np
from physics_experiments.HT_2d_ray_tracer import ray_tracer_2d
import matplotlib.pyplot as plt
from lib.plots_stuff import axisEqual3D
from scipy.spatial.transform import Rotation as R
import trimesh
import math
from lib.vectors import vector_normalize, norm, vector, origin
from matplotlib.patches import Rectangle
from math import floor, ceil


def plot_points(points,vertices=[]):
    X = []
    Y = []
    Z = []
    X_v = []
    Y_v = []
    Z_v = []
    for pp in points:
        X.append(pp[0])
        Y.append(pp[1])
        Z.append(pp[2])
    if vertices:
        for v in vertices:
            X_v.append(v[0])
            Y_v.append(v[1])
            Z_v.append(v[2])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs=X, ys=Y, zs=Z, marker=".")
    if vertices:
        ax.scatter(xs=X_v, ys=Y_v, zs=Z_v, marker="o", color = "r")
    #ax.plot(X, Y, zs=Z, marker=".", linestyle=None)
    axisEqual3D(ax)
    plt.show()

def vector_rotate(vector, rot_angle, rotation_axis=np.array([0, 0, 1])):
    """Rotate relative to oy;
    vector - which rotate;
    rot_angle - rotation angle in radians"""
    lv = norm(vector)
    vector = vector / lv
    rotation_vector = rot_angle * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    return np.array(rotation.apply(vector) * lv)

def plot_square(square_center, r_new, ax, **kwargs):
    ax.plot([square_center[0]-r_new, square_center[0]-r_new], [square_center[1]+r_new, square_center[1]-r_new], **kwargs)
    ax.plot([square_center[0]-r_new, square_center[0]+r_new], [square_center[1]-r_new, square_center[1]-r_new], **kwargs)
    ax.plot([square_center[0]+r_new, square_center[0]+r_new], [square_center[1]+r_new, square_center[1]-r_new], **kwargs)
    ax.plot([square_center[0]+r_new, square_center[0]-r_new], [square_center[1]+r_new, square_center[1]+r_new], **kwargs)

def transform_coords(coords: np.ndarray, MIP_level):
    s = coords / (2 ** MIP_level)
    return np.floor(s)

def voxelize_box(box, mip = 0):
    unit_sz = 2**mip
    for x in np.arange(floor(box[0][0]/unit_sz), floor(box[0][0]/unit_sz) + ceil(box[0][1]/unit_sz - box[0][0]/unit_sz)):
        for y in np.arange(floor(box[1][0]/unit_sz), floor(box[1][0]/unit_sz) + ceil(box[1][1]/unit_sz - box[1][0]/unit_sz)):
            for z in np.arange(floor(box[2][0]/unit_sz), floor(box[2][0]/unit_sz) + ceil(box[2][1]/unit_sz - box[2][0]/unit_sz)):
                yield x, y, z

def clip_box(box1, box2):
    res = np.zeros([3,2], dtype=float)
    for i in range(3):
        dim1 = box1[i]
        dim2 = box2[i]
        res[i] = np.clip(dim2, dim1[0], dim1[1])
    return res
def find_children(ax, x,y,z,box1, box2, mip, lowest_mip=3):

    if mip >= lowest_mip:
        actual_number_of_voxels = 2 ** (mip-1)
        new_box = clip_box(box1, box2)
        for x,y,z in voxelize_box(new_box, mip=mip-1):

            find_children(ax,x,y,z,[[x*actual_number_of_voxels, x*actual_number_of_voxels+actual_number_of_voxels],
                          [y*actual_number_of_voxels, y*actual_number_of_voxels+actual_number_of_voxels],
                          [z*actual_number_of_voxels, z*actual_number_of_voxels+actual_number_of_voxels]], box2, mip-1)

    else:
        actual_number_of_voxels = 2 ** (mip)
        plot_square(np.array([x * actual_number_of_voxels + actual_number_of_voxels / 2,
                              y * actual_number_of_voxels + actual_number_of_voxels / 2,
                              z * actual_number_of_voxels + actual_number_of_voxels / 2]), actual_number_of_voxels / 2, ax,
                    color="y", linestyle="-")
        ax.add_patch(Rectangle((x * actual_number_of_voxels, y * actual_number_of_voxels), actual_number_of_voxels,
                               actual_number_of_voxels,
                               #                              edgecolor='pink',
                               facecolor='blue',
                               fill=True,
                               alpha=0.1,
                               lw=5))
        plt.axis("equal")
        aaaaa = 1


def calc(angle, cone_angle, ax):

    cone_angle = np.radians(cone_angle)

    angle_of_cone_rotation = np.radians(angle)

    d_0 = 1

    koeff = 5

    init_vect = np.array([0,1,0])
    init_vect =init_vect * koeff
    init_vect = vector_rotate(init_vect, angle_of_cone_rotation)

    init_vector_normalized = vector_normalize(init_vect)
    init_vect_right = vector_rotate(init_vect, -cone_angle)
    init_vect_left = vector_rotate(init_vect, cone_angle)


    plt.plot([0.5, 0.5 + init_vect[0]] , [0.5, 0.5 +  init_vect[1]] , "r", "-")

    plt.plot([0.5, 0.5 + init_vect_right[0]] , [0.5, 0.5 + init_vect_right[1]] , "b", "-")

    plt.plot([0.5, 0.5 + init_vect_left[0]] , [0.5, 0.5 + init_vect_left[1]] , "b", "-")

    r = d_0 * math.sin(cone_angle)


    TRACE_STEPS = 18

    if False:
        square_center = init_vector_normalized*d_0
        mip = max(np.log2(2 * r), 0)
        actual_number_of_voxels = 2**(np.ceil(mip))
        plot_square(square_center, r, ax, color ="k", linestyle = "-")
        plot_square(square_center, actual_number_of_voxels/2, ax, color ="y", linestyle = "-")
        circle1 = plt.Circle((square_center[0], square_center[1]), r, color='r', fill=False)
        ax.add_patch(circle1)


    last_r = r
    last_dist = d_0
    for m in range(0, TRACE_STEPS):
        print(m)
        # Size of the selection zone
        r_new = (last_r + last_dist ) * (np.sin(cone_angle)/(1-np.sin(cone_angle)))
        # place of center for next selection zone
        d_new =  last_r/1 + last_dist  + r_new

        # calculate the mip-scale of the selection box
        mip = max(np.log2(2 * r_new), 0)
        #mip = np.ceil(mip)
        mip = np.floor(mip)
        print("mip: ", mip)
        actual_number_of_voxels = 2 ** mip
        square_center = np.array([0.5,0.5,0.5]) + init_vector_normalized * d_new
        #circle1 = plt.Circle((square_center[0],square_center[1]),r_new,color='r', fill=False)
        # ax.add_patch(circle1)

        last_r = r_new
        last_dist = d_new

        if mip == 3:
            aaaaa = 1

        FULL_LOOKUP_BOX = [[square_center[0] - r_new, square_center[0] + r_new],
                           [square_center[1] - r_new, square_center[1] + r_new],
                           [square_center[2] - r_new, square_center[2] + r_new]]

        plot_square(square_center, r_new,
                    ax,
                    color="k", linestyle="-")

        count = 0
        for x,y,z in voxelize_box(FULL_LOOKUP_BOX, mip=mip):
            print(x,y,z)
            find_children(ax,x,y,z,[[x*actual_number_of_voxels, x*actual_number_of_voxels+actual_number_of_voxels],
                          [y*actual_number_of_voxels, y*actual_number_of_voxels+actual_number_of_voxels],
                          [z*actual_number_of_voxels, z*actual_number_of_voxels+actual_number_of_voxels]], FULL_LOOKUP_BOX,mip)
            print("count: ", count)
            count += 1




fig1, ax = plt.subplots()
#calc(70)
cone_angle = 9
num = int(360/(2*cone_angle))
for a in np.linspace(0, 360,1):
    calc(a, cone_angle, ax)



plt.axis("equal")
plt.xlabel("x, voxels")
plt.ylabel("y, voxels")
plt.grid()
plt.show()