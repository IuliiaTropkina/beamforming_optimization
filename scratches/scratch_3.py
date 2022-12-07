import pickle
import numpy as np
import matplotlib.pyplot as plt
import math

points_plot = False
grid_step = 0.1
wavelength = 1
P_Tx = 1
phase_const = 2 * math.pi / wavelength
ray_inc_length = 200*wavelength
ray_obs_length = ray_inc_length

ico_sub = 3
view_number = 7
number_an_az = 2
number_an_el = 2

path_output = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_UAV/article 2/RCS_3D/"
figure_name = f"plane_model1_lambda{wavelength}m_grid_step{grid_step}m_view{view_number}_3D_point_num{number_an_az*number_an_el}"
pickle_in = open(path_output + figure_name + ".pickle", "rb")
object_power = pickle.load(pickle_in)
angles_azimuth = np.array(object_power[0])
angles_elevation = np.array(object_power[1])
Poewr_dB_depending_on_Rx_coordinate = np.array(object_power[2])

angles_azimuth_mesh, angles_elevation_mesh = np.meshgrid(angles_azimuth, angles_elevation)

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(angles_azimuth_mesh, angles_elevation_mesh, Poewr_dB_depending_on_Rx_coordinate, cmap=plt.cm.YlGnBu_r)

#ax.plot_trisurf(angles_azimuth,angles_elevation,Poewr_dB_depending_on_Rx_coordinate)
#ax.set_zlim(0, 1)
#axisEqual3D(ax)
ax.set_xlabel("Azimuth angle, radians", fontname="Times New Roman") # fontsize=20
ax.set_ylabel("Elevation angle, radians", fontname="Times New Roman")
ax.set_zlabel("Power Rx, dB", fontname="Times New Roman")
# plt.xticks(fontsize=20, fontname="Times New Roman")
# plt.yticks(fontsize=20, fontname="Times New Roman")
plt.grid()
# plt.legend(['Object', 'Rx', 'Tx'])
plt.show()
