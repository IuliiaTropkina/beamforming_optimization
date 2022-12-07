from http import HTTPStatus

from twisted.web import server, resource
from twisted.internet import reactor, endpoints, threads
from twisted.web.static import File
from dataclasses import dataclass
import numpy as np
import time
import matplotlib.pyplot as plt
from math import sin, cos, pi, acos
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
    return np.acos((2 * d ** 2 - a ** 2) / (
            2 * d ** 2))


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r

def get_directions_from_vertices(vertices_file_in_name):
    object = trimesh.load(vertices_file_in_name)
    vertices = object.vertices
    for ind_v, v in enumerate(vertices):
        vertices[ind_v] = v/norm(v)
    return(vertices)

vertices_norm_for_directions = get_directions_from_vertices("C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Projects/voxel_engine/draft_engine/ico_sphere_sub5.obj")

engine = Engine()
colors, object_voxels, transparent, RX, TX = ray_tracer_2d.process_image("C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Projects/voxel_engine/draft_engine/test_img8.png")
print(len(object_voxels))
engine.IN.object_voxels = object_voxels
engine.IN.voxel_transparent = transparent
engine.IN.RX = RX
engine.IN.TX = TX
time_sec1 = time.time()
engine.start_calculations_3D_sphere_coord_test()
time_sec2 = time.time()
print("Time: ", time_sec2 - time_sec1)
# coord_in = [4,5,6]
# # cube =
# a = np.zeros((10,10,10))
# a[[3,4,5],[4,5,6],[5,6,7]] = 1

#azimuth_angles = np.linspace[]
indexes = [-3,0,-5]
# print("Spherical coordinates: ", np.array(cart2sph(indexes[0], indexes[1], indexes[2]))[0]*180/math.pi, np.array(cart2sph(indexes[0], indexes[1], indexes[2]))[1]*180/math.pi, np.array(cart2sph(indexes[0], indexes[1], indexes[2]))[2])
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #indexes = np.where(a)
# ax.scatter(indexes[0], indexes[1], indexes[2], marker=".")
# axisEqual3D(ax)
plt.show()





# initial_vect = np.array([1,1,1])
# initial_vect = initial_vect/norm(initial_vect)
# angles = np.linspace(0,19,20)*math.pi/180
# vectors = np.zeros((len(angles),3))
# vectors_diff = np.zeros((len(angles),3))
# for a_ind, a in enumerate(angles):
#     vectors[a_ind] = vector_rotate(initial_vect,a)
#     vectors_diff[a_ind] = vectors[a_ind] - initial_vect
#
# print(angles )
# print(vectors_diff)

#first: test_img.png MIP = 1; t = 0.21 sec
#first: test_img8.png MIP = 1; t = 0.369
#first: test_img5.png MIP = 1; t = 0.361
#first: test_img5.png MIP = 0; t = 0.325
#first: test_img5.png MIP = 2; t = 0.49