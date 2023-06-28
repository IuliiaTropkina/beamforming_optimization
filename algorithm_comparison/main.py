# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import trimesh
import matplotlib.pyplot as plt
import sys
from scipy import spatial
from datetime import datetime
import os
import math
import random
import json
from math import sqrt
from scipy.io import loadmat
import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random
import copy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras

# from rl.agents import DQNAgent
# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory

#TODO: 1) learing rate 2) number of layers/nuerons 3)activation function (relu, linear, sigmoid) 4) metric (accuracy, m..) 5)loss functio (adam,...)

def E2Power(E, Freq):
    c = 299792458  # [m / sec]
    Lambda = c / Freq
    #Power = (Lambda ** 2 * (np.abs(E)) ** 2) / (4 * math.pi)
    Power = np.abs(E)**2
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


def run_synthetic():
    # SYNTHETIC DATASET
    number_of_experiments = 7
    ARMS_NUMBER_RANDOM = 42
    ITER_NUMBER_RANDOM = 50000
    folder_name_figures_synthetic = f"scenario_synthetic_dataset"
    figures_path_syn = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Project_materials/beamforming/FIGURES/{folder_name_figures_synthetic}/"

    scenario_synthetic = ["depends", "not_pepends"]
    algorithm_names = ["EPS_greedy",
                       "UCB",
                       "THS"]
    parameters_syn = [[0.05, 0.1, 0.15],
                      [0.2, 0.5, 1],
                      [0.2, 0.5]]
    for sc in scenario_synthetic:
        cumulative_average_all = np.zeros(ITER_NUMBER_RANDOM)
        if sc == "depends":
            data_with_context = True
        else:
            data_with_context = False
        for alg_name, pars in zip(algorithm_names, parameters_syn):
            print(alg_name)
            for p in pars:
                for num in range(number_of_experiments):
                    print(num)
                    bandit_syn = Contextual_bandit(alg_name, ARMS_NUMBER_RANDOM, ITER_NUMBER_RANDOM, initial_mean=p, data_random=True,
                    random_data_with_CONTEXT=data_with_context)
                    cumulative_average, reward = bandit_syn.run_bandit()
                    cumulative_average_all += cumulative_average
                cumulative_average_all = cumulative_average_all / number_of_experiments
                pickle.dump(cumulative_average_all, open(
                    f"{figures_path_syn}/cumulative_average_{sc}_{alg_name}_arms{int(ARMS_NUMBER_RANDOM)}_{p}.pickle",
                    'wb'))


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-2)
    return dqn

def build_model(states, actions):
    bias = keras.initializers.Constant(value=0.0)
    model = Sequential()

    model.add(Dense(64, activation='relu', bias_initializer=bias, input_shape=states))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(actions, activation='linear'))


    return model


def plot_bandit(figure_name, y, label, y_label, title, marker, log_mode=True):
    plt.figure(figure_name)
    plt.xscale('linear')
    plt.plot(y, marker, label=label)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    if UPDATE_FIGURES_WITH_RESULT:
        plt.savefig(
            f"{figures_path}{figure_name}_{ARMS_NUMBER_RANDOM}arms.png",
            dpi=700, bbox_inches='tight')
    if log_mode:
        plt.xscale('log')
        if UPDATE_FIGURES_WITH_RESULT:
            plt.savefig(
                f"{figures_path}{figure_name}_{ARMS_NUMBER_RANDOM}arms_log.png",
                dpi=700, bbox_inches='tight')
    plt.grid()


def choose_random_no_context(arm_num):
    means = np.linspace(0, 1, ARMS_NUMBER_RANDOM)

    return np.random.randn() + means[arm_num]
    # return np.random.randn() + 3


# Choose a random action
def choose_random(arm_num, frame_num):
    point_of_max = int((frame_num / 1000) % ARMS_NUMBER_RANDOM)
    means = np.roll(np.hamming(ARMS_NUMBER_RANDOM), point_of_max)
    return np.clip(np.random.randn() + means[arm_num], 0, 1)
 # return np.random.randn() + 3
def find_angle_between_vectors(v1, v2):
    return math.acos((v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (norm(v1)*norm(v2))) #radians






# def transform_reward(min_limit_power, max_limit_power, min_limit_reward, max_limit_reward):


# def bin_rays_by_direction(beam_dirs, ray_dirs, e_field) -> dict:
#     dirs_sorted = {}
#     for ray_num, ray in enumerate(ray_dirs):
#         index_nearest_RX = spatial.KDTree(beam_dirs).query(ray)[1]
#         try:
#             dirs_sorted[index_nearest_RX].append(ray_num)
#         except:
#             dirs_sorted[index_nearest_RX] = [ray_num]
#
#     dirs_sorted_power = np.zeros(len(beam_dirs))
#
#     max_power = max(np.abs(e_field)**2)
#     for i in range(0, len(beam_dirs)):
#         e_for_dir = []
#         try:
#             dirs = dirs_sorted[i]
#             for ind in dirs:
#                 e_for_dir.append(e_field[ind])
#             #dirs_sorted_power[i] = max(power_for_dir)
#
#             if max(np.abs(e_for_dir)**2) == max_power:
#                 dirs_sorted_power[i] = max_power
#             else:
#                 dirs_sorted_power[i] = np.abs(sum(10*e_for_dir))**2
#         except:
#             dirs_sorted_power[i] = 0
#     # dirs_sorted_power = { k:max(v) for k,v in dirs_sorted.items()}
#     return dirs_sorted_power



class CIR_cache:
    def __init__(self, PATH, num_rt_frames_total, frames_per_data_frame=1000):
        self.num_rt_frames_total = num_rt_frames_total
        self.frames_per_data_frame = frames_per_data_frame
        self.all_rewards = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.all_rewards_normalized = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.max_reward = 0
        self.dirs_sorted_power = np.zeros((num_rt_frames_total, len(beam_directions)))

    def get_power(self, frame_number):
        power = np.zeros(ARMS_NUMBER_CIR)

        dir = TX_locations[frame_number] - RX_locations[frame_number]

        beam_number_nearest = spatial.KDTree(beam_directions).query(dir)[1]
        # print(f"dir {dir}, {beam_number_nearest}")
        angle = find_angle_between_vectors(beam_directions[beam_number_nearest], dir) #radians
        antenna_gain = self.antenna_pattern_3D[90+int(np.round(angle*180/math.pi)),int(np.round(angle*180/math.pi))]
        # print(f"antenna gain, {frame_number}, {antenna_gain}dBi, {10**(antenna_gain/10)}")
        dist = norm(dir)
        c = 299792458
        power[beam_number_nearest] = ((c/carrier_frequency) / (4 * math.pi * dist)) ** 2 * 10**(antenna_gain/10)
        power[0] = (((c/carrier_frequency) / (4 * math.pi * dist)) ** 2   * 10**(antenna_gain/10) )/20
        power[1] = (((c / carrier_frequency) / (4 * math.pi * dist) )** 2 * 10 ** (antenna_gain / 10) )/ 8

        return power


    def get_power_based_on_dataset(self, frame_num, ray_dirs, P):

        max_power = max(P)
        dirs_sorted = {}
        for ray_num, ray in enumerate(ray_dirs):
            index_nearest_RX = spatial.KDTree(beam_directions).query(ray)[1]
            try:
                dirs_sorted[index_nearest_RX].append(ray_num)
            except:
                dirs_sorted[index_nearest_RX] = [ray_num]



        for i in range(0, len(beam_directions)):
            power_for_dir = []
            indexes = []
            try:
                dirs = dirs_sorted[i]
                for ind in dirs:
                    power_for_dir.append(P[ind])
                    indexes.append(ind)
                ray_direction_for_antenna = ray_dirs[indexes[np.argmax(np.array(power_for_dir))]]
                angle = find_angle_between_vectors(beam_directions[i], ray_direction_for_antenna)
                antenna_gain = self.antenna_pattern_3D[
                    90 + int(np.round(angle * 180 / math.pi)), int(np.round(angle * 180 / math.pi))]

                self.dirs_sorted_power[frame_num, i] = max(power_for_dir) * 10 ** (antenna_gain / 10)

                if max(power_for_dir) != max_power:
                    self.dirs_sorted_power[frame_num, i] = self.dirs_sorted_power[i] * 10

            except:
                self.dirs_sorted_power[frame_num, i] = 0


    def get_all_rewards(self):
        antenna_data = loadmat(f'antenna_pattern28GHz_type{ANTENNA_TYPE}.mat')
        self.antenna_pattern_3D = antenna_data['a']
        for frame_num in range(self.num_rt_frames_total):
            file_name = f"{PATH}/CIR_scene_frame{frame_num + 1}_grid_step{grid_step}_voxel_size{voxel_size}_freq{carrier_frequency}"
            data = pickle.load(open(f"{file_name}.pickle", "rb"))
            directions_of_arrival_RX = data[0]
            directions_of_arrival_RX = np.array(directions_of_arrival_RX)
            directions_of_arrival_RX_for_antenna = - directions_of_arrival_RX
            E = data[2]  # v/m
            E = np.array(E)
            time_array = data[1]
            Power = E2Power(E, carrier_frequency)
            #d = bin_rays_by_direction(beam_directions, directions_of_arrival_RX_for_antenna, Power)
            #d = bin_rays_by_direction(beam_directions, directions_of_arrival_RX_for_antenna, E)
            self.get_power_based_on_dataset(frame_num, directions_of_arrival_RX_for_antenna, Power)

        for it_num in range(ITER_NUMBER_CIR):
            if it_num % 100 == 0:
                print(f"Reward is calculated {it_num/ITER_NUMBER_CIR *100 }")
            data_frame_num1 = it_num // self.frames_per_data_frame
            data_frame_num2 = data_frame_num1 + 1
            # data1 = self.binned_rays_by_frame[data_frame_num1] + self.get_noise()

            #data1 = self.binned_rays_by_frame[data_frame_num1]
            data1 = self.dirs_sorted_power[data_frame_num1]



            #data1 = self.get_power(data_frame_num1)

            for ar_num in range(ARMS_NUMBER_CIR):

                if data_frame_num1 == FRAME_NUMBER - 1:
                    self.all_rewards[ar_num, it_num] = data1[ar_num]

                else:
                    # data2 = self.binned_rays_by_frame[data_frame_num2] + self.get_noise()
                    #data2 = self.binned_rays_by_frame[data_frame_num2]
                    #data2 = self.get_power(data_frame_num2)
                    data2 = self.dirs_sorted_power[data_frame_num2]
                    Power = self.get_reward(data_frame_num1, it_num, data1[ar_num], data2[ar_num])
                    self.all_rewards[ar_num, it_num] = Power


        self.max_reward = np.max(self.all_rewards)
        self.all_rewards = self.all_rewards/self.max_reward
        #self.all_rewards = self.all_rewards

        #For DL testing??
        # self.all_rewards = np.zeros(np.shape(self.all_rewards_normalized))
        # self.all_rewards[np.where(self.all_rewards_normalized == 1.0)] = 1

        # plt.figure()
        # plt.hist(self.all_rewards.flatten(), bins=10)
        # plt.show(block=True)


    def plot_all_rewards(self):
        fig5 = plt.figure()
        plt.imshow(self.all_rewards[:, 0:ITER_NUMBER_CIR - 1:1000], aspect="auto")
        col_bar2 = plt.colorbar()
        col_bar2.set_label('Power, W')
        # plt.show()
        plt.xlabel("Iteration (every 5000)")
        plt.ylabel("Arm nubmer")
        # plt.xticks(fontname="Times New Roman", fontsize="16")
        # plt.yticks(fontname="Times New Roman", fontsize="16")

        plt.savefig(f"{figures_path}/additional_inform/rewards{ARMS_NUMBER_CIR}.png", dpi=700,
                    bbox_inches='tight')

        # fig6 = plt.figure()
        # plt.imshow(self.all_rewards_dBm[:, 0:ITER_NUMBER_CIR - 1:1000], aspect="auto")
        # col_bar3 = plt.colorbar()
        # col_bar3.set_label('Power, dBm')
        # # plt.show()
        # plt.xlabel("Iteration (every 1000)")
        # plt.ylabel("Arm nubmer")
        # # plt.xticks(fontname="Times New Roman", fontsize="16")
        # # plt.yticks(fontname="Times New Roman", fontsize="16")
        #
        # plt.savefig(f"{figures_path}/additional_inform/rewards_dBm_{ARMS_NUMBER_CIR}.png", dpi=700,
        #             bbox_inches='tight')

    def get_reward(self,  data_frame_num1, it_num, data1, data2):

        d = it_num / self.frames_per_data_frame - data_frame_num1
        if d == 0:
            val = data1
        else:
            v1 = data1
            v2 = data2
            val = v1 * (1 - d) + v2 * d


        # TODO: apply transform

        return val


    def get_noise(self):
        return np.random.randn()

    def choose_context_number(self, context, i_number):

        data_frame_num1 = i_number // self.frames_per_data_frame
        context_vector = vector_normalize(TX_locations[data_frame_num1] - RX_locations[data_frame_num1])
        context_number = spatial.KDTree(context).query(context_vector)[1]

        return context_number

    def choose_context(self, i_number, context_type):

        data_frame_num1 = i_number // self.frames_per_data_frame


        if context_type == "DOA":
            context_vector = vector_normalize(TX_locations[data_frame_num1] - RX_locations[data_frame_num1])
            context_vector_rounded = context_vector
        elif context_type == "location":
            context_vector =  TX_locations[data_frame_num1] - RX_locations[data_frame_num1]
            context_vector_rounded = np.round(context_vector / LOCATION_GRID_STEP)



        return context_vector_rounded
class MultipathChannel(Env):
    def __init__(self, number_of_actions, number_of_states, context_set, starting_point = 0):
        self.action_space = Discrete(number_of_actions)
        self.observation_space = Box(low=np.array([0]), high=np.array([number_of_states-1]))
        self.starting_point = starting_point
        self.context_set = context_set
        self.iter_count = self.starting_point
        # self.state = cir_cache.choose_context_number(self.context_set, self.starting_point)
        self.state = self.iter_count
        self.record_rewards = []
        self.record_actions = []
        self.record_states = []

    def step(self, action):
        if self.iter_count < np.shape(cir_cache.all_rewards)[1]:
            #self.state = cir_cache.choose_context_number(self.context_set, self.iter_count)
            self.state = self.iter_count
        else:
            self.state = self.reset()

        reward = cir_cache.all_rewards[action, int(i % np.size(cir_cache.all_rewards, 1))]
        self.iter_count += 1
        self.record_rewards.append(reward)
        self.record_actions.append(action)
        self.record_states.append(self.state)
        return self.state, reward, False, {}


    def reset(self):
        # self.record_rewards = []
        # self.record_actions = []
        # self.record_states = []
        # self.state = cir_cache.choose_context_number(self.context_set, self.starting_point)
        self.iter_count = self.starting_point
        self.state = self.iter_count
        return self.state


class Contextual_bandit:
    def __init__(self, alg_name, arms_number, iter_number, param, context_type, number_of_frames_between_SB_burst, interval_between_SB_in_iterations , interval_feedback_iter, last_part_of_frame_iter, context_set=[[]], data_random=False,
                 random_data_with_CONTEXT=False):
        self.interval_between_SB_in_iterations = interval_between_SB_in_iterations
        self.last_part_of_frame_iter = last_part_of_frame_iter
        self.interval_feedback_iter = interval_feedback_iter
        self.number_of_frames_between_SB_burst = number_of_frames_between_SB_burst
        self.context_type = context_type
        self.data_random = data_random
        self.random_data_with_CONTEXT = random_data_with_CONTEXT
        self.MAB = []
        self.arm_num = 0
        self.arm_SSB = 0
        self.existing_contexts = np.array([])
        self.context_set = context_set
        self.iter_number = iter_number
        self.context_for_all_iterations = np.zeros((len(context_set), ITER_NUMBER_CIR))
        self.chosen_beam_number = []
        self.all_rewards = np.zeros((len(context_set), ITER_NUMBER_CIR))
        self.param = param
        self.arms_number = arms_number
        self.alg_name = alg_name
        self.reward_exploitation = []
        self.exploitation_iterations = []
        self.reward_exploiration = []
        self.context_number_exploration = []
        self.selected_arms = []
        if len(context_set)!=0:
            if alg_name == "UCB":
                for _ in self.context_set:
                    self.MAB.append(UCB(arms_number, param))
            elif alg_name == "EPS_greedy":
                for _ in self.context_set:
                    self.MAB.append(EPS_greedy(arms_number, param))
            elif alg_name == "THS":
                for _ in self.context_set:
                    self.MAB.append(ThompsonSampling(arms_number, param))

    def add_context_space(self):
        if self.alg_name == "UCB":
            self.MAB.append(UCB(self.arms_number, self.param))
        elif self.alg_name == "EPS_greedy":
            self.MAB.append(EPS_greedy(self.arms_number, self.param))
        elif self.alg_name == "THS":
            self.MAB.append(ThompsonSampling(self.arms_number, self.param))

    def add_context(self,context):
        try:
            context_number = np.where(np.all(self.existing_contexts == context, axis=1))[0][0]
        except:
            self.existing_contexts = np.append(self.existing_contexts, np.array([context]), axis=0)
            context_number = len(self.existing_contexts) - 1
            self.add_context_space()
        return context_number

    def run_bandit(self):
        rewards = np.empty(self.iter_number)

        for i in range(self.iter_number):
            if self.data_random:
                context_number = 0
            else:
                context = cir_cache.choose_context(int(i % np.size(cir_cache.all_rewards, 1)), self.context_type)
                if self.context_type == "location":
                    if i == 0:
                        self.existing_contexts = np.array([context])
                        context_number = 0
                        self.add_context_space()
                    else:
                        context_number = self.add_context(context)
                elif self.context_type == "DOA":
                    context_number = cir_cache.choose_context_number(self.context_set, int(i % np.size(cir_cache.all_rewards, 1)))
            if REAL_PROTOCOL:
                iter_from_begining_of_frame = i % (iter_per_frame * self.number_of_frames_between_SB_burst)
                IS_DL = is_DL(iter_from_begining_of_frame, iter_per_DL)
                if IS_DL and (iter_from_begining_of_frame < iter_per_frame):

                    if is_SSB_start(iter_from_begining_of_frame, dur_SB_in_iterations,
                                    self.interval_between_SB_in_iterations, self.last_part_of_frame_iter):


                        self.arm_num = self.MAB[context_number].get_arm()
                        self.arm_SSB = copy.copy(self.arm_num)
                        obtained_reward = cir_cache.all_rewards[self.arm_num, int(i % np.size(cir_cache.all_rewards, 1))]
                        self.reward_exploiration.append(obtained_reward)
                        self.context_number_exploration.append(context_number)
                        self.exploitation_iterations.append(i)
                        self.selected_arms.append(self.arm_num)
                        self.MAB[context_number].update(self.arm_num, obtained_reward)
                        self.MAB[context_number].all_iter_count += 1

                    elif not is_SB(iter_from_begining_of_frame, dur_SB_in_iterations,
                                   self.interval_between_SB_in_iterations, self.last_part_of_frame_iter):
                        self.arm_num = copy.copy(self.MAB[context_number].arm_exploitation)

                    else:
                        # try:
                        #     self.arm_num = copy.copy(self.MAB[context_number].arm_exploration)
                        # except:
                        #     self.arm_num = copy.copy(self.MAB[context_number].arm_exploitation)

                        self.arm_num = copy.copy(self.arm_SSB)
                        self.exploitation_iterations.append(i)

                else:
                    self.arm_num = copy.copy(self.MAB[context_number].arm_exploitation)



                if not IS_DL and (iter_from_begining_of_frame < iter_per_frame):
                    if is_feedback(iter_from_begining_of_frame, iter_per_DL, self.interval_feedback_iter):
                        # for r, c, b in zip(self.reward_exploiration, self.context_number_exploration, self.selected_arms):

                        self.MAB[context_number].update_arm_exploitation()
                        self.arm_num = copy.copy(self.MAB[context_number].arm_exploitation)


            else:
                self.arm_num = self.MAB[context_number].get_arm()
            self.chosen_beam_number.append(self.arm_num)

            obtained_reward = cir_cache.all_rewards[self.arm_num, int(i % np.size(cir_cache.all_rewards, 1))]

            # self.MAB[context_number].update(self.arm_num, obtained_reward)



            # for the plot
            rewards[i] = obtained_reward
            # if self.MAB[context_number].EXPLOITATION:
            #     self.reward_exploitation.append(obtained_reward)
            #     self.exploitation_iterations.append(i)

        return rewards, self.exploitation_iterations


class UCB:
    def __init__(self, arms_number, c):
        self.arms_mean_reward = np.zeros(arms_number)
        self.arms_iter_count = np.ones(arms_number)
        self.c = c
        self.arms_number = arms_number
        self.all_iter_count = 0
        self.arm_exploitation = 0
    # Update the action-value estimate
    def update(self, arm_num, obtained_reward):
        self.arms_iter_count[arm_num] += 1
        self.arms_mean_reward[arm_num] = (1 - 1.0 / self.arms_iter_count[arm_num]) * self.arms_mean_reward[
            arm_num] + 1.0 / self.arms_iter_count[arm_num] * obtained_reward
    def update_arm_exploitation(self):
        self.arm_exploitation = np.argmax(self.arms_mean_reward)
    def get_arm(self):
        self.arm_exploitation = np.argmax(self.arms_mean_reward + self.c * np.sqrt(
            (np.log(self.all_iter_count)) / self.arms_iter_count))
        return self.arm_exploitation



class ThompsonSampling:
    def __init__(self, arms_number, initial_presision=0.0001, initial_mean=1 / 2):
        self.arms_mean_reward = np.zeros(arms_number)
        self.arms_iter_count = np.ones(arms_number)
        self.mu = np.full(arms_number, initial_mean)
        self.arms_number = arms_number
        self.pres = np.full(arms_number, initial_presision)
        self.all_iter_count = 0

        self.arm_exploitation = 0
    def update(self, arm_num, obtained_reward):
        self.arms_iter_count[arm_num] += 1
        self.arms_mean_reward[arm_num] = (1 - 1.0 / self.arms_iter_count[arm_num]) * self.arms_mean_reward[
            arm_num] + 1.0 / self.arms_iter_count[arm_num] * obtained_reward

        self.pres[arm_num] += self.arms_iter_count[arm_num]
        self.mu[arm_num] = ((self.pres[arm_num] * self.mu[arm_num]) + (
                    self.arms_mean_reward[arm_num] * self.arms_iter_count[arm_num])) / (
                                       self.pres[arm_num] + self.arms_iter_count[arm_num])

    def get_arm(self):
        samples = [np.random.normal(loc=self.mu[a], scale=1 / np.sqrt(self.pres[a])) for a in range(self.arms_number)]
        self.arm_exploitation = np.argmax(samples)

        return self.arm_exploitation


class EPS_greedy:
    def __init__(self, arms_number, eps):

        self.arms_mean_reward = np.zeros(arms_number)
        self.arms_iter_count = np.zeros(arms_number)
        self.eps = eps
        self.arms_number = arms_number
        self.all_iter_count = 0
        self.EXPLOITATION = False
        self.arm_exploitation = 0
        self.arm_exploration = 0
    def update(self, arm_num, obtained_reward):
        self.arms_iter_count[arm_num] += 1
        self.arms_mean_reward[arm_num] = (1 - 1.0 / self.arms_iter_count[arm_num]) * self.arms_mean_reward[
            arm_num] + 1.0 / self.arms_iter_count[arm_num] * obtained_reward

    def update_arm_exploitation(self):
        self.arm_exploitation = np.argmax(self.arms_mean_reward)
    def get_arm(self):
        p = np.random.random()
        if p < self.eps:
            self.EXPLOITATION = False
            self.arm_exploration = np.random.choice(self.arms_number)
            return self.arm_exploration
        else:
            self.EXPLOITATION = True
            self.arm_exploitation = np.argmax(self.arms_mean_reward)
            return self.arm_exploitation

def is_DL(iter_from_begining_of_frame, iter_per_DL):
    if iter_from_begining_of_frame < iter_per_DL:
        return True
    return False
def is_SSB_start(iter_from_begining_of_frame, dur_SB_in_iterations, interval_between_SB_in_iterations, last_part_of_frame_iter):
    if iter_from_begining_of_frame % (dur_SB_in_iterations+ interval_between_SB_in_iterations) == 0 and (iter_from_begining_of_frame < iter_per_DL - last_part_of_frame_iter):

        return True
    return False

def is_feedback(iter_from_begining_of_frame, iter_per_DL, interval_feedback_iter):
    if iter_from_begining_of_frame == iter_per_DL + interval_feedback_iter - 1:
        return True
    return False


def is_SB(iter_from_begining_of_frame, dur_SB_in_iterations, interval_between_SB_in_iterations, last_part_of_frame_iter):
    if iter_from_begining_of_frame % (dur_SB_in_iterations+ interval_between_SB_in_iterations) < dur_SB_in_iterations and (iter_from_begining_of_frame < iter_per_DL - last_part_of_frame_iter):
        return True
    return False





def sequential_search( number_of_frames_between_SB_burst, interval_between_SB_in_iterations , interval_feedback_iter,number_of_SB_in_burst, last_part_of_frame_iter):

    SEARCH = True
    max_reward_search = np.zeros(ARMS_NUMBER_CIR)
    beam_number_count = 0
    chosen_max_beam_number = 0
    threshold = 0
    trying_beam_number = 0
    sequential_search_reward = []
    chosen_beam_number_seq_search = []
    sequential_search_exploitation_itarations = []
    search_true = np.zeros(ITER_NUMBER_CIR)
    #chosen_beam_number_seq_search = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
    search_false = np.zeros(ITER_NUMBER_CIR)
    threshold_all = []
    iter_threshold = []
    for i in range(0, ITER_NUMBER_CIR):
        iter_from_begining_of_frame = i % (iter_per_frame * number_of_frames_between_SB_burst)
        IS_DL = is_DL(iter_from_begining_of_frame, iter_per_DL)

        if SEARCH:
            if IS_DL and (iter_from_begining_of_frame < iter_per_frame):

                if is_SSB_start(iter_from_begining_of_frame, dur_SB_in_iterations, interval_between_SB_in_iterations, last_part_of_frame_iter):


                    trying_beam_number = copy.copy(beam_number_count)
                    chosen_reward = cir_cache.all_rewards[trying_beam_number, int(i % np.size(cir_cache.all_rewards, 1))]
                    max_reward_search[beam_number_count] = chosen_reward
                    sequential_search_exploitation_itarations.append(i)

                    beam_number_count += 1

                elif not is_SB(iter_from_begining_of_frame, dur_SB_in_iterations, interval_between_SB_in_iterations,last_part_of_frame_iter):
                    trying_beam_number = copy.copy(chosen_max_beam_number)
                else:
                    trying_beam_number = copy.copy(beam_number_count)
                    trying_beam_number = trying_beam_number - 1
                    sequential_search_exploitation_itarations.append(i)

            else:
                trying_beam_number = copy.copy(chosen_max_beam_number)

        else:
            trying_beam_number = copy.copy(chosen_max_beam_number)

        if not IS_DL and (iter_from_begining_of_frame < iter_per_frame) :
            if is_feedback(iter_from_begining_of_frame, iter_per_DL, interval_feedback_iter):


                chosen_max_beam_number = np.argmax(max_reward_search)
                trying_beam_number = copy.copy(chosen_max_beam_number)


        chosen_reward = cir_cache.all_rewards[trying_beam_number, int(i % np.size(cir_cache.all_rewards, 1))]
        sequential_search_reward.append(chosen_reward)
        chosen_beam_number_seq_search.append(trying_beam_number)

        if beam_number_count == ARMS_NUMBER_CIR:
            threshold = max(max_reward_search) / 2
            threshold_all.append(threshold)
            iter_threshold.append(i)
            SEARCH = False
            search_false[i] = ARMS_NUMBER_CIR + 3
            beam_number_count = 0
        elif chosen_reward <= threshold and SEARCH == False:
            # if i < 1000:
            #     print(f"{i}, {i*duration_of_one_sample}, thresh {threshold}, chosen_reward {chosen_reward}, trying_beam_number {trying_beam_number}, chosen_max_beam_number {chosen_max_beam_number} ")
            search_true[i] = ARMS_NUMBER_CIR + 3
            SEARCH = True




    pickle.dump(sequential_search_reward,
                open(
                    f"{figures_path}/seq_search_reward_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                    'wb'))
    pickle.dump(threshold_all,
                open(
                    f"{figures_path}/threshold_all_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                    'wb'))
    pickle.dump(iter_threshold,
                open(
                    f"{figures_path}/iter_threshold_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                    'wb'))

    pickle.dump(search_true,
                open(
                    f"{figures_path}/search_true_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                    'wb'))


    pickle.dump(search_false,
                open(
                    f"{figures_path}/search_false_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                    'wb'))


    pickle.dump(sequential_search_exploitation_itarations,
                open(
                    f"{figures_path}/sequential_search_exploitation_itarations_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                    'wb'))
    pickle.dump(chosen_beam_number_seq_search,
                open(
                    f"{figures_path}/chosen_beam_number_seq_search_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                    'wb'))
    # pickle.dump(seq_search_exploitation_it_num,
    #             open(
    #                 f"{figures_path}/seq_search_exploitation_it_num_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{SSB_period}_consSSB{num_batch}.pickle",
    #                 'wb'))


if __name__ == '__main__':
    DUR_FRAME = 10e-3
    DUR_DL = 5e-3
    # SHIFT_FEEDBACK = 2.5e-3
    # DUR_FEEDBACK = 66.67e-6
    DUR_SB = 66.67e-6

    SCENARIO_DURATION = 8
    NUM_CYCLE = 30
    frames_per_data_frame = 10000
    FRAME_NUMBER = 38
    ITER_NUMBER_CIR = frames_per_data_frame * FRAME_NUMBER
    ITER_NUMBER_RANDOM = ITER_NUMBER_CIR



    duration_of_one_sample = SCENARIO_DURATION / ITER_NUMBER_CIR
    iter_per_frame = np.floor(DUR_FRAME / duration_of_one_sample)
    iter_per_DL = np.floor(DUR_DL/duration_of_one_sample)
    dur_SB_in_iterations = np.floor(DUR_SB/duration_of_one_sample)




    folder_name = sys.argv[2]
    seed_number = sys.argv[1]

    #
    np.random.seed(int(seed_number))
    # PLOTTING
    PLOT_ALL_REWARDS = False
    PLOT_CONTEXT = False
    PLOT_REWARDS_DESTRIBUTION = False
    SYNTHETIC = False


    if SYNTHETIC:
        run_synthetic()

    # DATASET FROM ENGINE
    voxel_size = 0.5
    grid_step = 0.1


    P_TX = 1
    carrier_frequency = 28e9

    LOCATION_GRID_STEP = 15



    SUBDIVISION = 4
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])

    ARMS_NUMBER_CIR = len(beam_directions)
    SUBDIVISION_2 = 2
    icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)
    ANTENNA_TYPE = 3


    NUMBERs_OF_CONS_SSB = np.array([64]) #[4,8,64]
    Numbers_of_frames_between_SSB = np.array([1,2,4,8,16]) #1,2,4,8,16
    REAL_PROTOCOL = True



    NUMBER_OF_ITERATIONS_TRAINING = ITER_NUMBER_CIR #250000
    # scenarios = ["uturn", "LOS_moving", "blockage"]
    scenarios = ["LOS"]
    #context_sets = [np.array(icosphere_context.vertices),np.array([[1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0]]), np.array([[1, 1, 0]])]
    #context_sets = [np.array(icosphere_context.vertices)]
    location_grid = []
    context_sets = [np.array(icosphere_context.vertices),location_grid]
    context_types = ["DOA"]
    # algorithm_names = ["EPS_greedy",
    #                    "UCB",
    #                    "THS"]
    cont_params = [len(np.array(icosphere_context.vertices)), LOCATION_GRID_STEP]
    folder_test = "real_protocol"
    algorithm_names = ["EPS_greedy"] #"DQL","EPS_greedy"
    # parameters = [[0.05, 0.1, 0.15],
    #               [10 ** (-7), 10 ** (-7) * 2, 10 ** (-7) / 2],
    #               [0.2, 0.5]]
    #parameters = [[0.01,0.02, 0.2, 0.5]]#UCB
    parameters = [[0.2, 0.4, 0.6, 0.7, 0.8, 0.9]]  # eps greedy 0.8


    def calc(number_of_frames_between_SB_burst,number_of_SB_in_burst):
        print(f"number_of_frames_between_SB_burst {number_of_frames_between_SB_burst}, number_of_SB_in_burst {number_of_SB_in_burst}")
        data_iterations = iter_per_DL - dur_SB_in_iterations * number_of_SB_in_burst
        interval_between_SB_in_iterations = np.floor(
            (data_iterations) / (number_of_SB_in_burst - 1))
        last_part_of_frame_iter = data_iterations - interval_between_SB_in_iterations * (number_of_SB_in_burst - 1)


        interval_feedback_iter = np.floor((iter_per_frame - iter_per_DL) / 2)

        sequential_search(number_of_frames_between_SB_burst, interval_between_SB_in_iterations , interval_feedback_iter,number_of_SB_in_burst, last_part_of_frame_iter)



        for con_set, con_type, cont_param in zip(context_sets,context_types, cont_params):

            # env = MultipathChannel( ARMS_NUMBER_CIR, len(con_set), con_set, starting_point = 0)
            # #env_test = MultipathChannel( ARMS_NUMBER_CIR, len(con_set), con_set, starting_point = NUMBER_OF_ITERATIONS_TRAINING)
            # env_test = MultipathChannel(ARMS_NUMBER_CIR, len(con_set), con_set,
            #                             starting_point=0)

            env = MultipathChannel(ARMS_NUMBER_CIR, ITER_NUMBER_CIR, con_set, starting_point = 0)
            #env_test = MultipathChannel( ARMS_NUMBER_CIR, len(con_set), con_set, starting_point = NUMBER_OF_ITERATIONS_TRAINING)
            env_test = MultipathChannel(ARMS_NUMBER_CIR, ITER_NUMBER_CIR, con_set,
                                        starting_point=0)
            if PLOT_CONTEXT:
                fig4 = plt.figure()
                ax = plt.axes(projection='3d')
                col = ["r", "b", "g", "m"]
                for b in beam_directions:
                    cont_number = spatial.KDTree(con_set).query(b)[1]
                    if len(con_set) == 4:
                        scatter_plot = ax.scatter3D(b[0], b[1], b[2], color=col[cont_number])
                        for ind_con, con in enumerate(con_set):
                            scatter_plot = ax.scatter3D(con[0] * 3, con[1] * 3, con[2] * 3, color=col[ind_con])
                            scatter_plot = ax.scatter3D(con[0] * 3, con[1] * 3, con[2] * 3, color=col[ind_con])
                    else:
                        scatter_plot = ax.scatter3D(b[0], b[1], b[2])
                        for ind_con, con in enumerate(con_set):
                            scatter_plot = ax.scatter3D(con[0] * 3, con[1] * 3, con[2] * 3)
                            scatter_plot = ax.scatter3D(con[0] * 3, con[1] * 3, con[2] * 3)



                fig = plt.figure()

                ax = plt.axes(projection='3d')

                scatter_plot = ax.scatter3D(beam_directions[:, 0], beam_directions[:, 1], beam_directions[:, 2])
                plt.savefig(f"{figures_path}/context.png", dpi=700, bbox_inches='tight')



            for alg_name, pars in zip(algorithm_names, parameters):
                print(alg_name)
                for p in pars:
                    if alg_name == "UCB" or alg_name == "EPS_greedy" or alg_name == "THS":
                        number_of_cycles = 1
                        bandit = Contextual_bandit(alg_name, ARMS_NUMBER_CIR, ITER_NUMBER_CIR, p, con_type, number_of_frames_between_SB_burst,interval_between_SB_in_iterations , interval_feedback_iter, last_part_of_frame_iter, context_set=con_set)
                        reward, exloitation_iterations  = bandit.run_bandit()

                        pickle.dump(len(bandit.existing_contexts), open(
                            f"{figures_path}/number_of_contexts_cont_par{cont_param}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                            'wb'))


                        pickle.dump(bandit.chosen_beam_number, open(
                            f"{figures_path}/chosen_arm_type{con_type}_context{len(con_set)}_{alg_name}_{p}_{ARMS_NUMBER_CIR}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pickle",
                            'wb'))



                    elif alg_name == "DQL":

                        states = env.observation_space.shape
                        actions = env.action_space.n

                        model = build_model(states, actions)

                        dqn = build_agent(model, actions)
                        # dqn.compile(Adam(lr=1e-3), metrics=['accuracy'])
                        dqn.compile(Adam(lr=1e-3), metrics=['accuracy'])

                        number_of_cycles = 1000
                        # for n_c in range(number_of_cycles):
                        #     env.reset()
                        dqn.fit(env, nb_steps=NUMBER_OF_ITERATIONS_TRAINING*number_of_cycles, visualize=False, verbose=1)

                        rewards_training = env.record_rewards
                        actions_training = env.record_actions
                        states_training = env.record_states

                        cum_rewards_training = np.cumsum(rewards_training) / (np.arange(len(rewards_training)) + 1)

                        pickle.dump(cum_rewards_training, open(
                            f"{figures_path}/training_reward_{alg_name}_context{len(con_set)}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}.pickle",
                            'wb'))

                        number_of_episodes = ITER_NUMBER_CIR # ITER_NUMBER_CIR-NUMBER_OF_ITERATIONS_TRAINING
                        results = dqn.test(env_test, nb_episodes=1, nb_max_episode_steps = number_of_episodes, visualize=False)
                        rewards_test = env_test.record_rewards
                        actions_test = env_test.record_actions
                        states_test = env_test.record_states

                        print(np.mean(results.history['episode_reward']))

                        all_rewards = rewards_test
                        all_actions = actions_test
                        all_states = states_test


                        # all_rewards = np.concatenate((rewards_training,rewards_test), axis = 0)
                        # all_actions = np.concatenate((actions_training,actions_test), axis = 0)
                        # all_states = np.concatenate((states_training, states_test), axis=0)

                        cumulative_average = np.cumsum(all_rewards) / (np.arange(len(all_rewards)) + 1)
                        actions_for_plot = np.zeros((ARMS_NUMBER_CIR, len(all_actions)))
                        for ac_num, ac in enumerate(all_actions):
                            actions_for_plot[ac, ac_num] = 1

                        states_for_plot = np.zeros((len(con_set), len(all_states)))
                        for ac_num, ac in enumerate(all_states):
                            states_for_plot[ac, ac_num] = 1


                        duration_of_one_sample = SCENARIO_DURATION / ITER_NUMBER_RANDOM  # 20 mcs 2e-5
                        fig_name3 = f"chosen_beam_context{len(con_set)}_{alg_name}_{p}_{ARMS_NUMBER_CIR}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}"
                        plt.figure(fig_name3)
                        its = np.linspace(0, ITER_NUMBER_CIR - 1, ITER_NUMBER_CIR)
                        plt.plot(its * duration_of_one_sample, actions_for_plot, ".")
                        plt.ylabel('Selected beam', fontsize=14)
                        plt.xlabel("Time, sec", fontsize=14)
                        # plt.yscale("log")
                        # plt.ylim(0,10)
                        plt.grid()
                        plt.legend(prop={'size': 12})
                        plt.yticks(fontsize=12)
                        plt.xticks(fontsize=12)
                        plt.savefig(
                            f"{figures_path}/additional_inform/{fig_name3}.png",
                            dpi=700, bbox_inches='tight')



                        fig21 = plt.figure()
                        plt.imshow(states_for_plot[:,0:len(all_actions) - 1:1000], aspect="auto")
                        plt.xlabel("Iteration (every 1000)")
                        plt.ylabel("State number")
                        # plt.xticks(fontname="Times New Roman", fontsize="16")
                        # plt.yticks(fontname="Times New Roman", fontsize="16")

                        plt.savefig(
                            f"{figures_path}/additional_inform/states_context{len(con_set)}_{alg_name}_{p}_{ARMS_NUMBER_CIR}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}.pdf",
                            dpi=700,
                            bbox_inches='tight')

                    pickle.dump(reward, open(
                        f"{figures_path}/reward_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}_seed{seed_number}.pickle",
                        'wb'))

                    pickle.dump(exloitation_iterations, open(
                        f"{figures_path}/exloitation_iterations_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}_seed{seed_number}.pickle",
                        'wb'))

                    # pickle.dump(reward_exploitation, open(
                    #     f"{figures_path}/reward_exploitation_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{number_of_frames_between_SB_burst}_consSSB{number_of_SB_in_burst}_seed{seed_number}.pickle",
                    #     'wb'))



    # folder_name_CIRS = f"CIRS_scenario_{sc}"
    # PATH = f"C:/Users/1.LAPTOP-1DGAKGFF/Desktop/Projects/voxel_engine/draft_engine/narvi/CIRS/{folder_name_CIRS}/"
    PATH = f"/home/hciutr/project_voxel_engine/voxel_engine/draft_engine/narvi/{folder_name}/CIRS"
    PATH_json = f"/home/hciutr/project_voxel_engine/voxel_engine/draft_engine/narvi/{folder_name}"
    RX_locations = []
    TX_locations = []
    for fr in range(1,50):
        with open(f"{PATH_json}/scene_frame{fr}.json") as json_file:
            info = json.load(json_file)
        RX_locations.append(info["RX_location"][0])
        TX_locations.append(info["TX_location"][0])
    TX_locations = np.array(TX_locations)

    RX_locations = np.array(RX_locations)

    figures_path = f"{PATH_json}/output_type{ANTENNA_TYPE}"


    try:
        os.makedirs(figures_path)
    except:
        print(f"Folder {figures_path} exists!")

    selected_beams_folder = f"{figures_path}/selected_beams"


    try:
        os.makedirs(selected_beams_folder)
    except:
        print(f"Folder {selected_beams_folder} exists!")

    try:
        os.makedirs(f"{figures_path}/additional_inform")
    except:
        print(f"Folder {figures_path} exists!")

    cir_cache = CIR_cache(PATH, FRAME_NUMBER, frames_per_data_frame=frames_per_data_frame)

    try:
        cir_cache.all_rewards = pickle.load(open(
            f"{PATH_json}/reward_antenna_type{ANTENNA_TYPE}_arms{int(ARMS_NUMBER_CIR)}_it{ITER_NUMBER_CIR}.pickle",
            "rb"))
        cir_cache.max_reward = pickle.load(open(
            f"{PATH_json}/max_reward_type{ANTENNA_TYPE}_arms{int(ARMS_NUMBER_CIR)}_it{ITER_NUMBER_CIR}.pickle",
            "rb"))
    except:
        cir_cache.get_all_rewards()
        pickle.dump(cir_cache.all_rewards, open(
            f"{PATH_json}/reward_antenna_type{ANTENNA_TYPE}_arms{int(ARMS_NUMBER_CIR)}_it{ITER_NUMBER_CIR}.pickle",
            'wb'))
        pickle.dump(np.array([cir_cache.max_reward]), open(
            f"{PATH_json}/max_reward_type{ANTENNA_TYPE}_arms{int(ARMS_NUMBER_CIR)}_it{ITER_NUMBER_CIR}.pickle",
            'wb'))

    ITER_NUMBER_CIR = ITER_NUMBER_CIR*NUM_CYCLE
    SCENARIO_DURATION = SCENARIO_DURATION* NUM_CYCLE

    if PLOT_ALL_REWARDS:
        cir_cache.plot_all_rewards()


        if PLOT_REWARDS_DESTRIBUTION:
            # beams_to_plot = [1,5,7,11]
            beams_to_plot = np.linspace(1, ARMS_NUMBER_CIR - 1, ARMS_NUMBER_CIR)
            binwidth = 100
            for i, b in enumerate(beams_to_plot):
                if sum(cir_cache.all_rewards[int(b), :]) != 0:
                    # Set up the plot
                    fig, ax = plt.subplots()

                    ax.hist(cir_cache.all_rewards[int(b), :], bins=int(binwidth), density=True,
                            color='blue', edgecolor='black')

                    # Title and labels
                    ax.set_title(f"Beam number {i}")
                    ax.set_xlabel('Power, W')
                    ax.set_ylabel("Number of times")
                    plt.grid()
                    plt.savefig(f"{figures_path}/beam_power_PDF_arms_num_{ARMS_NUMBER_CIR}_num{i}.pdf", dpi=700,
                                bbox_inches='tight')
                    # plt.hist([x1, x2, x3, x4, x5], bins=int(180 / 15), normed=True,
                    #          color=colors, label=names)





    oracle = []
    best_beam = np.zeros(ITER_NUMBER_CIR)
    for i in range(ITER_NUMBER_CIR):
        oracle.append(max(cir_cache.all_rewards[:, int(i % np.size(cir_cache.all_rewards, 1))]))
        best_beam[i] = np.argmax(cir_cache.all_rewards[:, int(i % np.size(cir_cache.all_rewards, 1))])

    pickle.dump(oracle, open(
        f"{figures_path}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle", 'wb'))
    pickle.dump(best_beam, open(
        f"{figures_path}/best_beam_arms{int(ARMS_NUMBER_CIR)}.pickle", 'wb'))



    pickle.dump(TX_locations, open(
        f"{figures_path}/TX_locations.pickle", 'wb'))
    pickle.dump(RX_locations, open(
        f"{figures_path}/RX_locations.pickle", 'wb'))
    for N_f in Numbers_of_frames_between_SSB:
        for n_b in NUMBERs_OF_CONS_SSB:

            calc(N_f,n_b)

