import itertools

import numpy as np
import matplotlib.pyplot as plt
from lib.stuff import RATIO2DB
from lib.numba_opt import numba_available, njit
from lib.vectors import norm, vector_normalize
import math
from math import sin, cos, pi
from scipy.spatial.transform import Rotation as R
import pickle
from http import HTTPStatus

import math
from lib.plots_stuff import axisEqual3D, plot_var_thick
from draft_engine.engine import Engine
import json
from lib.vectors import vector_normalize, norm, vector, origin
import requests
import numpy as np
from physics_experiments.HT_2d_ray_tracer import ray_tracer_2d
import matplotlib.pyplot as plt
from lib.plots_stuff import axisEqual3D
from scipy.spatial.transform import Rotation as R
import trimesh

def calculate_angle_cos_theorem(d, a):
    return math.acos((2 * d ** 2 - a ** 2) / (
            2 * d ** 2))

print(calculate_angle_cos_theorem(2500, 1)*180/math.pi)



print(numba_available)
def get_directions_from_vertices(vertices_file_in_name):
    object = trimesh.load(vertices_file_in_name)
    vertices = object.vertices
    for ind_v, v in enumerate(vertices):
        vertices[ind_v] = v/norm(v)
    return(vertices)


vertices_norm_for_directions = get_directions_from_vertices("C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Projects/voxel_engine/draft_engine/ico_sphere_sub8.obj")


def plot_points(points, ax, vertices=tuple(), s=None, **kwargs):

    ax.scatter(xs=points[:,0], ys=points[:,1], zs=points[:,2], s=s, marker = ".",  **kwargs)
    if vertices:
        ax.scatter(xs=vertices[:,0], ys=vertices[:,1], zs=vertices[:,2], s=s, marker="o",  **kwargs)
    #ax.plot(X, Y, zs=Z, marker=".", linestyle=None)
    axisEqual3D(ax)


yaxis = np.array([0, 1, 0])
def vector_rotate(vector, rot_angle, rotation_axis=yaxis):
    """Rotate relative to oy;
    vector - which rotate;
    rot_angle - rotation angle in radians"""
    lv = norm(vector)
    vector = vector / lv
    rotation_vector = rot_angle * rotation_axis
    rotation = R.from_rotvec(rotation_vector)
    return np.array(rotation.apply(vector) * lv)

# vec_array = []
# for a_az in np.linspace(0,2*math.pi, 180):
#     vector = np.array([1, 0, 0])
#     vector = vector_rotate(vector, a_az, rotation_axis=np.array([0, 0, 1]))
#     for a_el in np.linspace(0, math.pi, 90):
#         vec_array.append(vector_rotate(vector, a_el, rotation_axis=np.array([0, 1, 0])))
#
# vec_array = np.array(vec_array)


# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111, projection='3d')
# plot_points(vec_array, ax2, color="r", s=1)
# plt.show()

ax = np.array([1,0,0])
angles = np.linspace(0,2*math.pi, 360*2)
vectors = np.zeros((len(angles),3))
quantized_vectors = np.zeros((len(angles),3))

BITS = 7


def quantize_direction(dv:np.ndarray, bits:int=BITS)-> np.ndarray:
    ibase = 2**bits
    return np.round(dv * ibase)


points = {}

# for a_ind, a in enumerate(angles):
#     v =  vector_rotate(ax, a, rotation_axis=yaxis)
#     vectors[a_ind] = v
#     qv = quantize_direction(v)
#     BP1 = BITS  + 2
#     ibase = 2 ** BITS
#     kqv = np.array(qv + ibase, dtype=int)
#     assert (kqv >= 0).all()
#     k = kqv[0] + (kqv[1] << BP1) + (kqv[2] << BP1)
#     assert k < 2**63
#     if k not in points:
#         points[k] = qv



for p_ind, p in enumerate(vertices_norm_for_directions):
    if p[2]>0:
        qv = quantize_direction(p)
        BP1 = BITS  + 2
        ibase = 2 ** BITS
        kqv = np.array(qv + ibase, dtype=int)
        assert (kqv >= 0).all()
        k = kqv[0] + (kqv[1] << BP1) + (kqv[2] << (2*BP1))
        assert k < 2**63
        if k not in points:
            points[k] = qv


# vector = []
# vector2 = []
# for p_ind, p in enumerate(vertices_norm_for_directions):
#     if p[2]>0:
#         vector.append(p)
#         qv = quantize_direction(p)
#         vector2.append(vector_normalize(qv))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#plot_points(vector, "k", ax)
# plt.figure()
# plt.plot(vectors[:,0],vectors[:,2], ".", color = "k", markersize = 2)

qpoints = np.zeros([len(points),3])

i = itertools.count()
for k,v in points.items():
    qpoints[next(i)] = vector_normalize(v)

print(len(qpoints))
plot_points(qpoints, ax, color="r", s=1)
v1 = np.array([-0.0225, 0.0166, 0.4344])
v1 = v1/norm(v1)
v2 = np.array([-0.0102,0.0156,0.4341])
v2 = v2/norm(v2)

print(math.acos(np.dot(v1, v2))*180/math.pi)
color = iter("mbgc")
# for B2, d in [[BITS-1, vector(0,1,1)], [BITS-2, vector(1,1,1)], [BITS-3, vector(-1,1,1)], [BITS-4, vector(1,-1,1)]]:
#     region_dir = quantize_direction(vector_normalize(d), bits = B2)
#     rpoints = []
#     for k,v in points.items():
#         cv = quantize_direction(vector_normalize(v), bits = B2)
#         if (cv == region_dir).all():
#             rpoints.append(vector_normalize(v))
#
#     rpoints = np.stack(rpoints)
#     #print(rpoints)
#     plot_points(rpoints*1.01, ax, color=next(color), s=5)
#     # plt.plot(qpoints[:,0],qpoints[:,2], ".", color = "r", markersize = 2)
#     # plt.grid()
plt.show()
