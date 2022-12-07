from lib.numba_opt import jit_hardcore
from lib.transformations.quaternions import quaternion_from_axis_angle, rotate_vector, \
    quaternion_from_scalar_and_vector, quaternion_between_vectors, quaternion_multiply, quaternion_to_str
from lib.transformations.transformations import rotation_matrix
import numpy as np
import matplotlib.pyplot as plt
from lib.vectors import vector, vector_normalize, vector_project, norm
from lib.plots_stuff import plot_line
import pytest


R = rotation_matrix(np.pi, vector(0,0,1))
print(R)
R = R[0:3, 0:3]
print(R)
V = vector(1,1,0.1)
print(np.dot(R,V))


base_dir = vector(1,0,0)
base_E = vector(0,0,1)
base_B = vector(0,1,0)

dir = vector_normalize(vector(1,0.8,0.6))
print("Direction:", dir)
TX_orient = quaternion_between_vectors(base_dir, dir)
TX_orient2 = quaternion_multiply(TX_orient, quaternion_from_axis_angle(axis=base_dir, angle=np.pi/2))

def print_vec(v):
    """An informal, nicely printable string representation of the Quaternion object.
   """
    return "{:.3f}x {:+.3f}y {:+.3f}z".format(*v)
initial_point = vector(0,0,0)
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(projection='3d')
plot_line(ax, initial_point, initial_point + base_dir, color="b")
plot_line(ax, initial_point, initial_point + base_E, color="b", linestyle="-.")
plot_line(ax, initial_point, initial_point + base_B, color="b", linestyle="--")


initial_point_for_rotated = initial_point
for orient, c in zip([TX_orient, TX_orient2],["r","g"]):
    initial_point_for_rotated = initial_point_for_rotated + dir
    rotated_base_dir = rotate_vector(orient, base_dir)
    print("set: ", dir)
    print("rotated: ", rotated_base_dir)
    rotated_base_E = rotate_vector(orient, base_E)
    rotated_base_B = rotate_vector(orient, base_B)
    # print("Orientation quat:", quaternion_to_str(orient))
    # print("propagation direction:", print_vec(rotate_vector(orient, base_dir)))
    # print("E direction:", print_vec(rotate_vector(orient, base_E)))
    # print("B direction:", print_vec(rotate_vector(orient, base_B)))


    plot_line(ax, initial_point_for_rotated, initial_point_for_rotated + dir, color = c)
    plot_line(ax, initial_point_for_rotated, initial_point_for_rotated + rotated_base_E, color = c, linestyle="-.")
    plot_line(ax, initial_point_for_rotated, initial_point_for_rotated + rotated_base_B, color = c,linestyle="--")
plt.show()