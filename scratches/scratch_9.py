import numpy as np
from lib.vectors import vector_normalize, norm
import scipy.io
import trimesh
from scipy import spatial
import pickle
import matplotlib.pyplot as plt
from lib.plots_stuff import axisEqual3D
from phy_abstraction.antenna.antenna_makers import make_antenna_cellular
from phy_abstraction.antenna.antenna_plotters import plot_pattern_projection, plot_antenna_2D, plot_antenna_3D
from lib.transformations.quaternions import quaternion_from_euler
from lib.hexgrid import hexgrid_cells
from lib import rng, hexgrid
from scipy.spatial.transform import Rotation as R
from lib.numba_opt import jit_hardcore

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
def decart_to_spherical(x,y,z):
    if z==0:
        elev_angle = np.pi/2
    else:
        elev_angle = np.arctan(np.sqrt(x**2+y**2)/z)

    if x==0:
        azim_angle = np.pi/2
    else:
        azim_angle = np.arctan(y/x)
    return [elev_angle, azim_angle]

def get_RCS_from_table(RCS_table, directions,dir_from_Tx, dir_to_Rx):
    index_nearest_Tx = spatial.KDTree(directions).query(dir_from_Tx)[1]
    index_nearest_Rx = spatial.KDTree(directions).query(dir_to_Rx)[1]
    #plot_points([[0,0,0],directions[index_nearest_Tx]], vertices=[[0,0,0],dir_from_Tx])
    return RCS_table[index_nearest_Tx][index_nearest_Rx]


def get_directions(vertices_file_in_name: str)->np.ndarray:
    object = trimesh.load(vertices_file_in_name)
    vertices = object.vertices

    # angles = np.zeros((len(ico_sphere_vertices),2))
    # for vert_ind, vert in enumerate(ico_sphere_vertices):
    #     print(vert)
    #     angles[vert_ind] = decart_to_spherical( vert[0], vert[1], vert[2])
    vertices_norm = np.zeros((len(vertices), 3))
    for vert_ind, vert in enumerate(vertices):
        vertices_norm[vert_ind] = vector_normalize(-vert)

    return vertices_norm

#@jit_hardcore
def get_target_CIR():


    power_target =  np.full(50, fill_value=-190)
    tof_target = np.full(50, fill_value=-190)
    dir_of_departure_target =np.full((50,3), fill_value=-190)
    dir_of_arrival_target =np.full((50,3), fill_value=190)
    doppler_target = np.full(50, fill_value=-30)



    return dir_of_arrival_target, dir_of_departure_target, tof_target, doppler_target, power_target

speed_of_light = 299792458
carrier_freq = 300e6
grid_step = 0.004
wavelength = speed_of_light/carrier_freq


scene_name = "plane_model1"

file_input_RCS_UAV = f"{scene_name}_freq{carrier_freq}Hz_grid_step{grid_step}m_RCS.pickle"
file_input_RCS_building = f"building_12_30_freq{carrier_freq}Hz_grid_step0.04m_RCS.pickle"
#file_input = f"test.mat"

# Loading object


# RCS_mat = scipy.io.loadmat(file_input)
# RCS_table = np.array(RCS_mat["RCS"])



receiving_UAV_number = 1 #Drone number to be the Rx

height_bs = 30
radius_bs = 700
high_UAV_level = 65
low_UAV_level = 55
UAV_speed = 20
Rx_speed = UAV_speed
Power_Tx = 30
N_samples = 100000
N_UAV = 33
N_bs = 19
N_building = 50
antenna_downtill = 10

building_locations = np.zeros((3,N_building))
BS_locations = np.zeros((N_bs,3))
antenna_angles = np.zeros(N_bs)
DATA = np.zeros((N_samples,N_bs),dtype = dict)

#
cell_positions = hexgrid_cells(N_bs)

for i, c in enumerate(cell_positions):
    x, y, ang = hexgrid.hexgrid(c[0], c[1], radius_bs)
    BS_locations[i] = np.array([x,y,height_bs])
    antenna_angles[i] = ang[0]

building_locations[0] = np.random.uniform(size=N_building,low=min(BS_locations[:,0])-radius_bs, high=max(BS_locations[:,0])+radius_bs)
building_locations[1] = np.random.uniform(size=N_building,low=min(BS_locations[:,1])-radius_bs, high=max(BS_locations[:,1])+radius_bs)
building_locations[2] = np.full(N_building, fill_value=0)

initioal_UAV_direction = np.array([0,1,0])
axis_rotation_direction = np.array([0,0,1])
for Tx_loc_index, Tx_location in enumerate(BS_locations):
    print(Tx_loc_index)

    for sample_num in range(0,N_samples):


        dir_of_arrival_UAV, dir_of_departure_UAV, tof_UAV, doppler_UAV, power_UAV = get_target_CIR()

        dir_of_arrival_building, dir_of_departure_building, tof_building, doppler_building, power_building = get_target_CIR()


        DATA[sample_num, Tx_loc_index] = {"tof": np.concatenate((tof_UAV,tof_building),axis=0), "power_at_rx": np.concatenate((power_UAV,power_building),axis=0),
                                          "dir_of_departure": np.concatenate((dir_of_departure_UAV,dir_of_departure_building),axis=0),
                                          "dir_of_arrival": np.concatenate((dir_of_arrival_UAV,dir_of_arrival_building),axis=0),"doppler_shift": np.concatenate((doppler_UAV,doppler_building)),
                                          "building_locations" : building_locations, "BS_location" : Tx_location }


scipy.io.savemat(f"DATA_scenario_1.mat", mdict={"DATA" : DATA, "parameters" : {
    "wavelength" :wavelength,
    "receiving_UAV_number": receiving_UAV_number,
    "height_bs" : height_bs,
    "radius_bs" : radius_bs,
    "high_UAV_level" : high_UAV_level,
    "low_UAV_level" : low_UAV_level,
    "UAV_speed" : Rx_speed,
    "Rx_speed" : Rx_speed,
    "Power_Tx" : Power_Tx,
    "N_samples" : N_samples,
    "N_UAV" : N_UAV,
    "N_bs": N_bs,
    "N_building" : N_building,
    "antenna_downtill" : antenna_downtill,
    "grid_step":grid_step} })

