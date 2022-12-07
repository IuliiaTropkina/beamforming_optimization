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

def calc(angle, cone_angle, ax):
    cone_angle = np.radians(cone_angle)
    angle_of_cone_rotation = np.radians(angle)

    d_0 = 1
    koeff = 130

    init_vect = np.array([0,1,0])
    init_vect =init_vect * koeff
    init_vect = vector_rotate(init_vect, angle_of_cone_rotation)

    init_vector_normalized = vector_normalize(init_vect)
    init_vect_right = vector_rotate(init_vect, -cone_angle)
    init_vect_left = vector_rotate(init_vect, cone_angle)


    plt.plot([0.5, 0.5 + init_vect[0]] , [0.5, 0.5 +  init_vect[1]] , "b", "-")
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

    LUC = 0
    last_r = r
    last_dist = d_0
    for m in range(0, TRACE_STEPS):
        # Size of the selection zone
        r_new = (last_r + last_dist ) * (np.sin(cone_angle)/(1-np.sin(cone_angle)))
        # place of center for next selection zone
        #d_new = last_r + last_dist  + r_new
        d_new = last_r/2 + last_dist + r_new
        # calculate the mip-scale of the selection box
        mip = max(np.log2(2 * r_new), 0)
        #mip = np.ceil(mip)
        mip = np.floor(mip)

        actual_number_of_voxels = 2 ** mip
        square_center = np.array([0.5,0.5,0.5]) + init_vector_normalized * d_new
        plt.plot(square_center[0], square_center[1],".", color="r",   markersize = 7)
        circle1 = plt.Circle((square_center[0],square_center[1]),r_new,color='r', fill=False)
        ax.add_patch(circle1)

        last_r = r_new
        last_dist = d_new


        plot_square(square_center, r_new, ax, color="k", linestyle="-")
        #np.array([s_floor[0], s_ceil[0]]), np.array([s_floor[1], s_ceil[1]]), np.array([s_floor[2], s_ceil[2]])

        square_center_rounded = np.array([round(square_center[0]), round(square_center[1]), round(square_center[2])])
        s = 2 ** (mip)
        #square_center_rounded = transform_coords(square_center, mip)* s + s/2
        transformed_coords1 = transform_coords(square_center, mip)* s + s / 2
        diff = transformed_coords1 - square_center
        xs = [transformed_coords1[0]]
        ys = [transformed_coords1[1]]
        zs = [transformed_coords1[2]]
        if diff[0] > 0:
            xs.append(transformed_coords1[0] - s)
        elif diff[0] < 0:
            xs.append(transformed_coords1[0] + s)
        else:
            xs.append(transformed_coords1[0] - s)
            xs.append(transformed_coords1[0] + s)

        if diff[1] > 0:
            ys.append(transformed_coords1[1] - s)
        elif diff[1] < 0:
            ys.append(transformed_coords1[1] + s)
        else:
            ys.append(transformed_coords1[1] - s)
            ys.append(transformed_coords1[1] + s)

        if diff[2] > 0:
            zs.append(transformed_coords1[2] - s)
        elif diff[2] < 0:
            zs.append(transformed_coords1[2] + s)
        else:
            zs.append(transformed_coords1[2] - s)
            zs.append(transformed_coords1[2] + s)

        # for x in xs:
        #     for y in ys:
        #         for z in zs:
        #             plot_square(np.array([x,y,z]), actual_number_of_voxels/2, ax, color ="y", linestyle = "-")
        #
        #             ax.add_patch(Rectangle((x-s/2, y-s/2), s, s,
        #                                    #edgecolor='pink',
        #                                    facecolor='blue',
        #                                    fill=True,
        #                                    alpha=0.1,
        #                                    lw=5))




fig1, ax = plt.subplots()
#calc(70)
cone_angle = 9
num = int(360/(2*cone_angle))
# for a in np.linspace(0, 360,1):
#     calc(a, cone_angle, ax)
for a in [0, 30]:
    calc(a, cone_angle, ax)

plt.axis("equal")
plt.xlabel("x, voxels")
plt.ylabel("y, voxels")
plt.grid()
plt.show()