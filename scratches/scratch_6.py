import matplotlib.pyplot as plt
from lib.plots_stuff import axisEqual3D
import pickle
import numpy as np
import math
grid_step = 0.004
wavelength = 1
P_Tx = 1
phase_const = 2 * math.pi / wavelength
ray_inc_length = 200*wavelength
ray_obs_length = ray_inc_length
radius_for_view_plotting = 5 #m
ico_sub = 3
bin_len = 256
view_number = 7
a = 2320090
path_output = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_UAV/article 2/RCS_3D/"
#figure_name = f"sphere_rad3_sub4_lambda{wavelength}m_grid_step{grid_step}m_impulse_response_impulse_response_3"
figure_name = f"plane_model1_lambda1m_grid_step0.004m_bins{bin_len}_impulse_response_0"
pickle_in_num = open(path_output + figure_name + ".pickle", "rb")
imp_resp = pickle.load(pickle_in_num)
#imp_resp_dict = {}
# coeff = 1e9
# for i in range(1,len(imp_resp[0])):
#     if round(imp_resp[0][i]*coeff) in imp_resp_dict.keys():
#         imp_resp_dict[round(imp_resp[0][i]*coeff)] = imp_resp_dict[round(imp_resp[0][i]*coeff)] + imp_resp[1][i] + imp_resp[2][i] * 1j
#     else:
#         imp_resp_dict[round(imp_resp[0][i] * coeff)] = imp_resp[1][i] + imp_resp[2][i] * 1j
plt.figure(figsize=[15, 10])
plt.title(f"distance = {400} m, grid size = {grid_step} m, wavelength = {wavelength} m, bins = {bin_len}",fontname="Times New Roman", fontsize=20)
plt.plot(imp_resp[0], imp_resp[1], '>g')
plt.plot(imp_resp[0], imp_resp[2],'.r')
# plt.plot(list(imp_resp_dict.keys()), np.real(list(imp_resp_dict.values())), '>g')
# plt.plot(list(imp_resp_dict.keys()), np.imag(list(imp_resp_dict.values())),'.r')
#plt.plot(np.array(imp_resp[0])*coeff, np.array(imp_resp[1]),'.g')
plt.xlabel("delay, s", fontname="Times New Roman", fontsize=20)
plt.ylabel("Electric field", fontname="Times New Roman", fontsize=20)
plt.xticks(fontsize=20, fontname="Times New Roman")
plt.yticks(fontsize=20, fontname="Times New Roman")
plt.grid()
plt.legend(['Real', 'Imag'], prop={'family': "Times New Roman", "size": 20})
plt.savefig(path_output + figure_name + ".png", dpi=700, bbox_inches='tight')
# plt.ylim([-90,-45])


plt.show()

