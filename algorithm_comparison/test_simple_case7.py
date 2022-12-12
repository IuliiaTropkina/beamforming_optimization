import numpy as np
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import pickle
import os
import trimesh
from math import sqrt
import math
from scipy import spatial

def bin_rays_by_direction(beam_dirs, ray_dirs, power) -> dict:
    dirs_sorted = {}
    for ray_num, ray in enumerate(ray_dirs):
        index_nearest_RX = spatial.KDTree(beam_dirs).query(ray)[1]
        try:
            dirs_sorted[index_nearest_RX].append(ray_num)
        except:
            dirs_sorted[index_nearest_RX] = [ray_num]

    dirs_sorted_power = np.zeros(len(beam_dirs))
    for i in range(0, len(beam_dirs)):
        power_for_dir = []
        try:
            dirs = dirs_sorted[i]
            for ind in dirs:
                power_for_dir.append(power[ind])
            dirs_sorted_power[i] = max(power_for_dir)
        except:
            dirs_sorted_power[i] = 0
    # dirs_sorted_power = { k:max(v) for k,v in dirs_sorted.items()}
    return dirs_sorted_power

def E2Power(E, Freq):
    c = 299792458  # [m / sec]
    Lambda = c / Freq
    Power = (Lambda ** 2 * (np.abs(E)) ** 2) / (4 * math.pi)
    # Power = (Lambda**2*(np.abs(E))**2)/(4*math.pi*Z0)
    return Power


def norm(v: np.ndarray) -> float:
    assert v.ndim == 1
    # assert v.dtype != np.complex128
    return sqrt((v * v).sum())


def vector_normalize(v: np.ndarray) -> np.ndarray:
    n = norm(v)
    if n == 0:
        raise ValueError("Vector has null length")
    return v / norm(v)


class CIR_cache:
    def __init__(self, PATH, num_rt_frames_total, TX_locations, RX_locations, cont_set, frames_per_data_frame=1):
        self.binned_rays_by_frame = []
        self.frames_per_data_frame = frames_per_data_frame
        self.all_rewards = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.all_rewards_dBm = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.all_rewards_normalized = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.all_contexts = np.zeros((1, ITER_NUMBER_CIR))
        self.max_reward = 0
        self.TX_locations = TX_locations
        self.RX_locations = RX_locations
        self.context_set = cont_set
        for frame_num in range(num_rt_frames_total):
            file_name = f"{PATH}CIR_scene_frame{frame_num + 1}_grid_step{grid_step}_voxel_size{voxel_size}_freq{carrier_frequency}"
            data = pickle.load(open(f"{file_name}.pickle", "rb"))
            directions_of_arrival_RX = data[0]
            directions_of_arrival_RX = np.array(directions_of_arrival_RX)
            directions_of_arrival_RX_for_antenna = - directions_of_arrival_RX
            E = data[2]  # v/m
            E = np.array(E)
            time_array = data[1]
            Power = E2Power(E, carrier_frequency)
            d = bin_rays_by_direction(beam_directions, directions_of_arrival_RX_for_antenna, Power)
            self.binned_rays_by_frame.append(d)

    def get_all_rewards(self):
        for it_num in range(ITER_NUMBER_CIR):
            for ar_num in range(ARMS_NUMBER_CIR):
                self.all_rewards[ar_num, it_num] = self.get_reward(ar_num, it_num)
                if self.all_rewards[ar_num, it_num] != 0:
                    self.all_rewards_dBm[ar_num, it_num] = 10 * np.log10(
                        self.all_rewards[ar_num, it_num] / (10 ** (-3)))
                else:
                    self.all_rewards_dBm[ar_num, it_num] = -100
        #     self.all_rewards_normalized[:, it_num] = self.all_rewards[:, it_num]/np.max(self.all_rewards[:, it_num])
        #
        self.max_reward = np.max(self.all_rewards)
        self.all_rewards = self.all_rewards/self.max_reward
        #
        # self.all_rewards = np.zeros(np.shape(self.all_rewards_normalized))
        # self.all_rewards[np.where(self.all_rewards_normalized == 1.0)] = 1


    def get_all_contexts(self):
        for it_num in range(ITER_NUMBER_CIR):
            self.all_contexts[0, it_num] = self.choose_context_number(self.context_set, it_num)

    def get_reward(self, arm_num, it_num):
        data_frame_num1 = it_num // self.frames_per_data_frame
        data_frame_num2 = data_frame_num1 + 1
        # data1 = self.binned_rays_by_frame[data_frame_num1] + self.get_noise()

        data1 = self.binned_rays_by_frame[data_frame_num1]


        if data_frame_num1 == FRAME_NUMBER - 1:
            return data1[arm_num]
        # data2 = self.binned_rays_by_frame[data_frame_num2] + self.get_noise()
        data2 = self.binned_rays_by_frame[data_frame_num2]

        d = it_num / self.frames_per_data_frame - data_frame_num1
        if d == 0:
            val = data1[arm_num]
        else:
            v1 = data1[arm_num]
            v2 = data2[arm_num]
            val = v1 * (1 - d) + v2 * d

        # TODO: apply transform
        return val

    def get_noise(self):
        return np.random.randn()

    def choose_context_number(self, context, i_number):

        data_frame_num1 = i_number // self.frames_per_data_frame

        # arm_num_max = np.argmax(self.binned_rays_by_frame[data_frame_num1])
        # beam_dir_max_power = beam_directions[arm_num_max]
        # context_changed_direction = - context
        # context_number = spatial.KDTree(context_changed_direction).query(beam_dir_max_power)[1]


        direction_fromRX_toTX = vector_normalize(self.TX_locations[data_frame_num1] - self.RX_locations[data_frame_num1])
        context_number = spatial.KDTree(context).query(direction_fromRX_toTX)[1]
        return context_number



def agent(X_shape, Y_shape):
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_shape=X_shape, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(8, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(8, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(8, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(4, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(4, activation='relu', kernel_initializer=init))
    model.add(keras.layers.Dense(Y_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])
    return model

def create_features_matrix(cir_cache, number_of_samples_in_one_dataset, context_type = "DOA", datasettype="synthetic"):
    if datasettype == "channel":
        features = np.zeros((2, number_of_samples_in_one_dataset))
        cir_cache.get_all_contexts()
        features[0, : ] = cir_cache.all_contexts
        features[1, : ] = np.argmax(cir_cache.all_rewards, axis=0)
        features[1,:] = np.roll(features[1,:],1)
        if context_type == "DOA":
            return np.array([features[0, :]])
        elif context_type == "previous_beam":
            return features
        else:
            return np.array([features[1, :]])

    elif datasettype == "synthetic":
        features = np.linspace(0, number_of_samples_in_one_dataset - 1, number_of_samples_in_one_dataset)

        return features


def create_reward_matrix(cir_cache, number_of_outputs, number_of_samples_in_one_dataset, datasettype = "synthetic"):
    ALL_REWARD = np.zeros((number_of_outputs, number_of_samples_in_one_dataset))
    if datasettype == "channel":

        cir_cache.get_all_rewards()
        oracle = []
        for i in range(number_of_samples_in_one_dataset):
            oracle.append(max(cir_cache.all_rewards[:, i]))
        pickle.dump(oracle, open(f"{figures_path}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle", 'wb'))

        sequential_search_reward = []
        max_reward_search = np.zeros(ARMS_NUMBER_CIR)
        chosen_beam_number_seq_search = np.zeros((ARMS_NUMBER_CIR, number_of_samples_in_one_dataset))
        threshold = 0
        SEARCH = True
        iter_number_for_search = 0
        chosen_beam_number = iter_number_for_search
        chosen_reward = cir_cache.all_rewards[0, i]
        for i in range(0, number_of_samples_in_one_dataset):
            chosen_reward = cir_cache.all_rewards[chosen_beam_number, i]
            sequential_search_reward.append(chosen_reward)
            chosen_beam_number_seq_search[chosen_beam_number, i] = 1

            if SEARCH:
                max_reward_search[iter_number_for_search] = chosen_reward
                iter_number_for_search += 1
                chosen_beam_number += 1
                if iter_number_for_search == ARMS_NUMBER_CIR:
                    chosen_beam_number = np.argmax(max_reward_search)
                    threshold = max(max_reward_search) / 2
                    SEARCH = False
            else:
                if chosen_reward < threshold:
                    SEARCH = True
                    iter_number_for_search = 0
                    chosen_beam_number = 0

        pickle.dump(sequential_search_reward,
                    open(f"{figures_path}/sequential_search_arms{int(ARMS_NUMBER_CIR)}.pickle", 'wb'))


        ALL_REWARD = cir_cache.all_rewards[0:number_of_outputs, 0:number_of_samples_in_one_dataset]


    elif datasettype == "synthetic":

        ALL_REWARD[0, :np.shape(ALL_REWARD)[1] // 3] = 1
        ALL_REWARD[0, -np.shape(ALL_REWARD)[1] // 3:] = 1

        # ALL_REWARD[0] = np.random.randint(2, size=(1, len(ALL_REWARD[0])))

        # ALL_REWARD[1] = abs((ALL_REWARD[0] - 1))
        # ALL_REWARD[1, 0:int(np.shape(ALL_REWARD)[1] / 2)] = 20
    features = create_features_matrix(cir_cache, number_of_samples_in_one_dataset, datasettype=datasettype)
    return ALL_REWARD, features

if __name__ == '__main__':
    voxel_size = 0.5
    grid_step = 0.1
    P_TX = 1
    carrier_frequency = 900e6

    frames_per_data_frame = 1000 #10000
    FRAME_NUMBER = 38
    ITER_NUMBER_CIR = frames_per_data_frame * FRAME_NUMBER
    ITER_NUMBER_RANDOM = ITER_NUMBER_CIR

    SUBDIVISION = 0
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])


    ARMS_NUMBER_CIR = len(beam_directions)
    number_of_outputs = ARMS_NUMBER_CIR



    context_subdevisions = [3]
    context_sets = []
    for SUBDIVISION_2 in context_subdevisions:
        icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)
        context_sets.append(np.array(icosphere_context.vertices))



    number_of_one_dataset_to_be_repeated = 1
    number_of_samples_in_one_dataset = ITER_NUMBER_CIR
    total_number_of_samples = number_of_one_dataset_to_be_repeated * number_of_samples_in_one_dataset
    #features = np.linspace(0, number_of_samples_in_one_dataset - 1, number_of_samples_in_one_dataset)
    for cont_set in context_sets:

        folder_name_figures = f"scenario_uturn"
        figures_path = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/{folder_name_figures}/DL/"

        datasettype = "channel"

        RX_locations = []
        TX_locations = []
        sc = "uturn"
        folder_name_CIRS = f"CIRS_scenario_{sc}"
        PATH = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Projects/voxel_engine/draft_engine/narvi/CIRS/{folder_name_CIRS}/"
        for fr in range(1, 50):
            with open(f"{PATH}/scene_frame{fr}.json") as json_file:
                info = json.load(json_file)
            RX_locations.append(info["RX_location"][0])
            TX_locations.append(info["TX_location"][0])
        TX_locations = np.array(TX_locations)
        RX_locations = np.array(RX_locations)
        cir_cache = CIR_cache(PATH, FRAME_NUMBER, TX_locations, RX_locations, cont_set,
                              frames_per_data_frame=frames_per_data_frame)


        ALL_REWARD, features = create_reward_matrix(cir_cache, number_of_outputs, number_of_samples_in_one_dataset,
                                                    datasettype=datasettype)


        features_normalized = np.zeros(np.shape(features))
        number_of_features = len(features)
        for i in range(number_of_features):
            features_normalized[i] = features[i] / max(features[i])



        Y = []
        for _ in range(number_of_outputs):
            Y.append([])

        X = []
        for _ in range(number_of_features):
            X.append([])


        for n in range(number_of_one_dataset_to_be_repeated):
            for i in range(number_of_outputs):
                Y[i].append(ALL_REWARD[i])
            for ii in range(number_of_features):
                X[ii].append(features_normalized[ii])

        X = np.array(X)
        Y = np.array(Y)

        YY = np.zeros((number_of_outputs, total_number_of_samples))
        XX = np.zeros((number_of_features, total_number_of_samples))
        for i in range(number_of_outputs):
            YY[i] = np.reshape(Y[i], (1, total_number_of_samples))
        for ii in range(number_of_features):
            XX[ii] = np.reshape(X[ii], (1, total_number_of_samples))

        rewards_predicted = np.zeros((number_of_outputs, total_number_of_samples))
        batch_size = 1
        for out_num in range(number_of_outputs):
            model = agent((number_of_features, 1), 1)
            model.fit(XX.transpose(), np.array([YY[out_num]]).transpose(), batch_size=batch_size, verbose=0, shuffle=True)

            rew = model.predict(features_normalized.transpose())
            rewards_predicted[out_num] = rew[:,0,0]


        test_name = "many_beams_dir_plus_previous"
        pickle.dump(rewards_predicted,
                    open(f"{figures_path}/{test_name}_rewards_predicted_con_num{len(cont_set)}_arms{int(ARMS_NUMBER_CIR)}.pickle", 'wb'))

        pickle.dump(ALL_REWARD,
                    open(f"{figures_path}/{test_name}_rewards_con_num{len(cont_set)}_arms{int(ARMS_NUMBER_CIR)}.pickle", 'wb'))

        pickle.dump(features,
                    open(f"{figures_path}/{test_name}_features_con_num{len(cont_set)}.pickle", 'wb'))





