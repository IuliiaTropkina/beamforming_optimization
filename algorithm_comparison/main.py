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

import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random

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


# def transform_reward(min_limit_power, max_limit_power, min_limit_reward, max_limit_reward):


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


class CIR_cache:
    def __init__(self, PATH, num_rt_frames_total, frames_per_data_frame=1000):
        self.binned_rays_by_frame = []
        self.frames_per_data_frame = frames_per_data_frame
        self.all_rewards = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.all_rewards_dBm = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.all_rewards_normalized = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.max_reward = 0

        for frame_num in range(num_rt_frames_total):
            file_name = f"{PATH}/CIR_scene_frame{frame_num + 1}_grid_step{grid_step}_voxel_size{voxel_size}_freq{carrier_frequency}"
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
            #Normalization
            #self.all_rewards_normalized[:, it_num] = self.all_rewards[:, it_num]/np.max(self.all_rewards[:, it_num])
            self.all_rewards_normalized[:, it_num] = self.all_rewards[:, it_num]

        self.all_rewards = self.all_rewards_normalized
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

        plt.savefig(f"{figures_path}rewards{ARMS_NUMBER_CIR}.png", dpi=700,
                    bbox_inches='tight')

        fig6 = plt.figure()
        plt.imshow(self.all_rewards_dBm[:, 0:ITER_NUMBER_CIR - 1:1000], aspect="auto")
        col_bar3 = plt.colorbar()
        col_bar3.set_label('Power, dBm')
        # plt.show()
        plt.xlabel("Iteration (every 1000)")
        plt.ylabel("Arm nubmer")
        # plt.xticks(fontname="Times New Roman", fontsize="16")
        # plt.yticks(fontname="Times New Roman", fontsize="16")

        plt.savefig(f"{figures_path}rewards_dBm_{ARMS_NUMBER_CIR}.png", dpi=700,
                    bbox_inches='tight')

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

        reward = cir_cache.all_rewards[action, self.iter_count]
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
    def __init__(self, alg_name, arms_number, iter_number, param, context_type, SSB_period, num_batch, context_set=[[]], data_random=False,
                 random_data_with_CONTEXT=False):
        self.context_type = context_type
        self.data_random = data_random
        self.random_data_with_CONTEXT = random_data_with_CONTEXT
        self.MAB = []
        self.arm_num = 0
        self.existing_contexts = np.array([])
        self.context_set = context_set
        self.iter_number = iter_number
        self.context_for_all_iterations = np.zeros((len(context_set), ITER_NUMBER_CIR))
        self.chosen_beam_number = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        self.all_rewards = np.zeros((len(context_set), ITER_NUMBER_CIR))
        self.param = param
        self.arms_number = arms_number
        self.alg_name = alg_name
        self.reward_exploitation = []
        self.exploitation_iterations = []
        self.SSB_period = SSB_period
        self.num_batch = num_batch

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
                context = cir_cache.choose_context(i, self.context_type)
                if self.context_type == "location":
                    if i == 0:
                        self.existing_contexts = np.array([context])
                        context_number = 0
                        self.add_context_space()
                    else:
                        context_number = self.add_context(context)
                elif self.context_type == "DOA":
                    context_number = cir_cache.choose_context_number(self.context_set, i)
            if REAL_PROTOCOL:
                if is_SSB(i, self.SSB_period,self.num_batch):
                    self.arm_num = self.MAB[context_number].get_arm()
                else:
                    self.arm_num = self.MAB[context_number].arm_exploitation

            else:
                self.arm_num = self.MAB[context_number].get_arm()

            self.chosen_beam_number[self.arm_num, i] = 1
            #self.context_for_all_iterations[context_number, i] = 1
            if self.data_random:
                if self.random_data_with_CONTEXT:
                    obtained_reward = choose_random(self.arm_num, i)
                else:
                    obtained_reward = choose_random_no_context(self.arm_num)
            else:
                obtained_reward = cir_cache.all_rewards[self.arm_num, i]

            if REAL_PROTOCOL:
                if is_SSB(i,  self.SSB_period,self.num_batch):
                    self.MAB[context_number].update(self.arm_num, obtained_reward)
            else:
                self.MAB[context_number].update(self.arm_num, obtained_reward)


            # for the plot
            rewards[i] = obtained_reward
            # if self.MAB[context_number].EXPLOITATION:
            #     self.reward_exploitation.append(obtained_reward)
            #     self.exploitation_iterations.append(i)

            if not is_SSB(i,  self.SSB_period,self.num_batch):
                self.reward_exploitation.append(obtained_reward)
                self.exploitation_iterations.append(i)

            self.MAB[context_number].all_iter_count += 1
        cumulative_average = np.cumsum(rewards) / (np.arange(self.iter_number) + 1)

        return cumulative_average, rewards, self.reward_exploitation, self.exploitation_iterations


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
    def update(self, arm_num, obtained_reward):
        self.arms_iter_count[arm_num] += 1
        self.arms_mean_reward[arm_num] = (1 - 1.0 / self.arms_iter_count[arm_num]) * self.arms_mean_reward[
            arm_num] + 1.0 / self.arms_iter_count[arm_num] * obtained_reward

    def get_arm(self):
        p = np.random.random()
        if p < self.eps:
            self.EXPLOITATION = False
            return np.random.choice(self.arms_number)
        else:
            self.EXPLOITATION = True
            self.arm_exploitation = np.argmax(self.arms_mean_reward)
            return self.arm_exploitation


if __name__ == '__main__':
    folder_name = sys.argv[2]
    seed_number = sys.argv[1]
    #
    # np.random.seed(int(seed_number))
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

    frames_per_data_frame = 10000 #10000
    FRAME_NUMBER = 38
    ITER_NUMBER_CIR = frames_per_data_frame * FRAME_NUMBER
    ITER_NUMBER_RANDOM = ITER_NUMBER_CIR

    SUBDIVISION = 2
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    beam_directions = np.array(icosphere.vertices)
    #beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])

    ARMS_NUMBER_CIR = len(beam_directions)
    SUBDIVISION_2 = 2
    icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)


    SCENARIO_DURATION = 8
    NUMBERs_OF_CONS_SSB = np.array([4,8,64])
    SSB_periods = np.array([5,10,20,40,80,160])
    SSB_periods = SSB_periods*10**(-3)
    REAL_PROTOCOL = True
    def is_SSB(iteration, SSB_period, num_batch):

        number_of_periods = SCENARIO_DURATION/SSB_period

        duration_of_one_sample = SCENARIO_DURATION/ITER_NUMBER_RANDOM

        number_of_iterations_between_cons_SSB = int((5*10**(-3)/num_batch)/(duration_of_one_sample))
        max_number_of_iteration_in_set = int((num_batch-1) * number_of_iterations_between_cons_SSB)
        number_of_iterations_per_one_SSB_period = int(SSB_period/duration_of_one_sample)
        if (iteration % number_of_iterations_per_one_SSB_period) % number_of_iterations_between_cons_SSB == 0 and (iteration % number_of_iterations_per_one_SSB_period) <= max_number_of_iteration_in_set:
            return True
        return False


    NUMBER_OF_ITERATIONS_TRAINING = ITER_NUMBER_CIR #250000
    # scenarios = ["uturn", "LOS_moving", "blockage"]
    scenarios = ["LOS"]
    #context_sets = [np.array(icosphere_context.vertices),np.array([[1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0]]), np.array([[1, 1, 0]])]
    #context_sets = [np.array(icosphere_context.vertices)]
    location_grid = []
    context_sets = [np.array(icosphere_context.vertices),location_grid]
    context_types = ["DOA", "location"]
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
    parameters = [[0.15, 0.05, 0.4, 0.8, 0.95]]  # eps greedy


    def calc(SSB_period,num_batch):


        data = np.zeros((1,ITER_NUMBER_RANDOM))
        for i in range(ITER_NUMBER_RANDOM):
            if is_SSB(i, SSB_period, num_batch):
                data[0,i] = 1


        fig7777 = plt.figure()
        plt.imshow(data[:,0:int(ITER_NUMBER_CIR / 16)], aspect="auto")
        plt.xlabel("Data type")
        plt.ylabel("Iteration")
        # plt.xticks(fontname="Times New Roman", fontsize="16")
        # plt.yticks(fontname="Times New Roman", fontsize="16")

        plt.savefig(f"{selected_beams_folder}/packet_SSB_period{SSB_period}_cons_period{num_batch}.pdf",
                    dpi=700, bbox_inches='tight')

        sequential_search_reward = []
        seq_search_exploitation_reward = []
        seq_search_exploitation_it_num = []

        max_reward_search = np.zeros(ARMS_NUMBER_CIR)
        chosen_beam_number_seq_search = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
        threshold = 0
        SEARCH = True
        iter_number_for_search = 0
        chosen_reward = cir_cache.all_rewards[0, i]
        sequential_search_time = []
        eps_greedy_time = []
        num_it_search = 0
        num_it_eps_greedy = 0
        eps = 0.15
        exp_vs_expl = False
        trying_beam_number = 0
        for i in range(0, ITER_NUMBER_CIR):
            if SEARCH and not REAL_PROTOCOL:
                trying_beam_number = iter_number_for_search
            elif SEARCH and REAL_PROTOCOL and is_SSB(i,SSB_period, num_batch):
                trying_beam_number = iter_number_for_search
            else:
                trying_beam_number = np.argmax(max_reward_search)

            if exp_vs_expl:
                eps_greedy_time.append(num_it_eps_greedy / ((i + 1) - num_it_eps_greedy))
            else:
                eps_greedy_time.append(num_it_eps_greedy / ((i + 1)))

            p = np.random.random()
            if p < eps:
                num_it_eps_greedy += 1

            chosen_reward = cir_cache.all_rewards[trying_beam_number, i]
            sequential_search_reward.append(chosen_reward)
            chosen_beam_number_seq_search[trying_beam_number, i] = 1
            if exp_vs_expl:
                sequential_search_time.append(num_it_search/((i+1)-num_it_search))
            else:
                sequential_search_time.append(num_it_search / ((i + 1)))

            if REAL_PROTOCOL:
                SEARCH_IN_SSB = SEARCH and is_SSB(i,SSB_period,num_batch)

            else:
                SEARCH_IN_SSB = SEARCH

            if SEARCH_IN_SSB:
                max_reward_search[iter_number_for_search] = chosen_reward
                iter_number_for_search += 1
                num_it_search += 1
                if iter_number_for_search == ARMS_NUMBER_CIR:

                    threshold = max(max_reward_search) / 2
                    SEARCH = False
                    iter_number_for_search = 0
            else:
                seq_search_exploitation_reward.append(chosen_reward)
                seq_search_exploitation_it_num.append(i)
                if chosen_reward < threshold:
                    SEARCH = True





        fig75 = plt.figure()
        plt.imshow(chosen_beam_number_seq_search[:, 0:int(ITER_NUMBER_CIR)], aspect="auto")
        plt.xlabel("Iteration")
        plt.ylabel("Beam nubmer")
        # plt.xticks(fontname="Times New Roman", fontsize="16")
        # plt.yticks(fontname="Times New Roman", fontsize="16")

        plt.savefig(f"{selected_beams_folder}/chosen_arm_sequantial_search_{ARMS_NUMBER_CIR}_SSBperiod{SSB_period}_consSSB{num_batch}.pdf",
                    dpi=700,
                    bbox_inches='tight')


        avarage_sequential_search = np.cumsum(sequential_search_reward) / (np.arange(ITER_NUMBER_CIR) + 1)


        avarage_sequential_search_dBm = 10 * np.log10(avarage_sequential_search / (10 ** (-3)))
        pickle.dump(avarage_sequential_search, open(f"{figures_path}/cumulative_avarage_sequential_search_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{SSB_period}_consSSB{num_batch}.pickle", 'wb'))
        pickle.dump(seq_search_exploitation_reward,
                    open(f"{figures_path}/seq_search_exploitation_reward_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{SSB_period}_consSSB{num_batch}.pickle",
                         'wb'))
        pickle.dump(seq_search_exploitation_it_num,
                    open(f"{figures_path}/seq_search_exploitation_it_num_arms{int(ARMS_NUMBER_CIR)}_SSBperiod{SSB_period}_consSSB{num_batch}.pickle",
                         'wb'))


        pickle.dump(sequential_search_time,
                    open(f"{figures_path}/exp_expl_time_sequential_search_arms{int(ARMS_NUMBER_CIR)}_eps{eps}.pickle",
                         'wb'))
        pickle.dump(eps_greedy_time,
                    open(f"{figures_path}/exp_expl_time_eps_greedy_arms{int(ARMS_NUMBER_CIR)}_eps{eps}.pickle",
                         'wb'))
        random_choice = []
        for i in range(ITER_NUMBER_CIR):
            random_num = np.random.choice(ARMS_NUMBER_CIR)
            random_choice.append(cir_cache.all_rewards[random_num, i])
        avarage_random_choice = np.cumsum(random_choice) / (np.arange(ITER_NUMBER_CIR) + 1)
        pickle.dump(avarage_random_choice, open(
            f"{figures_path}/cumulative_avarage_random_choice_arms{int(ARMS_NUMBER_CIR)}.pickle", 'wb'))

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
                        bandit = Contextual_bandit(alg_name, ARMS_NUMBER_CIR, ITER_NUMBER_CIR, p, con_type, SSB_period, num_batch, context_set=con_set)
                        cumulative_average, reward, reward_exploitation, exloitation_iterations  = bandit.run_bandit()

                        pickle.dump(len(bandit.existing_contexts), open(
                            f"{figures_path}/number_of_contexts_cont_par{cont_param}_SSBperiod{SSB_period}_consSSB{num_batch}.pickle",
                            'wb'))

                        fig7 = plt.figure()
                        plt.imshow(bandit.chosen_beam_number[:, 0:int(ITER_NUMBER_CIR/16)], aspect="auto")
                        plt.xlabel("Iteration")
                        plt.ylabel("Beam nubmer")
                        # plt.xticks(fontname="Times New Roman", fontsize="16")
                        # plt.yticks(fontname="Times New Roman", fontsize="16")

                        plt.savefig(
                            f"{selected_beams_folder}/chosen_arm_type{con_type}_context{len(con_set)}_{alg_name}_{p}_{ARMS_NUMBER_CIR}_SSBperiod{SSB_period}_consSSB{num_batch}.pdf",
                            dpi=700,
                            bbox_inches='tight')

                        # fig89 = plt.figure()
                        # plt.imshow(bandit.context_for_all_iterations[:, 0:ITER_NUMBER_CIR - 1:1000], aspect="auto")
                        # plt.xlabel("Iteration (every 1000)", fontname="Times New Roman", fontsize="16")
                        # plt.ylabel("Context nubmer", fontname="Times New Roman", fontsize="16")
                        # plt.xticks(fontname="Times New Roman", fontsize="16")
                        # plt.yticks(fontname="Times New Roman", fontsize="16")
                        #
                        # plt.savefig(
                        #     f"{selected_beams_folder}/context_for_all_iterations_cont_type{con_type}_context{len(con_set)}.pdf",
                        #     dpi=700,
                        #     bbox_inches='tight')

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

                        fig20 = plt.figure()
                        plt.imshow(actions_for_plot[:,0:len(all_actions) - 1:1000], aspect="auto")
                        plt.xlabel("Iteration (every 1000)")
                        plt.ylabel("Beam nubmer") #, fontname="Times New Roman", fontsize="16"
                        # plt.xticks(fontname="Times New Roman", fontsize="16")
                        # plt.yticks(fontname="Times New Roman", fontsize="16")

                        plt.savefig(
                            f"{selected_beams_folder}/chosen_arm_context{len(con_set)}_{alg_name}_{p}_{ARMS_NUMBER_CIR}_SSBperiod{SSB_period}_consSSB{num_batch}.pdf",
                            dpi=700,
                            bbox_inches='tight')

                        fig21 = plt.figure()
                        plt.imshow(states_for_plot[:,0:len(all_actions) - 1:1000], aspect="auto")
                        plt.xlabel("Iteration (every 1000)")
                        plt.ylabel("State number")
                        # plt.xticks(fontname="Times New Roman", fontsize="16")
                        # plt.yticks(fontname="Times New Roman", fontsize="16")

                        plt.savefig(
                            f"{selected_beams_folder}/states_context{len(con_set)}_{alg_name}_{p}_{ARMS_NUMBER_CIR}_SSBperiod{SSB_period}_consSSB{num_batch}.pdf",
                            dpi=700,
                            bbox_inches='tight')

                    pickle.dump(cumulative_average, open(
                        f"{figures_path}/cumulative_average_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{SSB_period}_consSSB{num_batch}_seed{seed_number}.pickle",
                        'wb'))

                    pickle.dump(exloitation_iterations, open(
                        f"{figures_path}/exloitation_iterations_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{SSB_period}_consSSB{num_batch}_seed{seed_number}.pickle",
                        'wb'))

                    pickle.dump(reward_exploitation, open(
                        f"{figures_path}/reward_exploitation_bandit_{alg_name}_cont_type{con_type}_cont_param{cont_param}_arms{int(ARMS_NUMBER_CIR)}_{p}_num_cycle{number_of_cycles}_SSBperiod{SSB_period}_consSSB{num_batch}_seed{seed_number}.pickle",
                        'wb'))



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
    figures_path = f"{PATH_json}/output"
    try:
        os.makedirs(figures_path)
    except:
        print(f"Folder {figures_path} exists!")

    selected_beams_folder = f"{figures_path}/selected_beams"


    try:
        os.makedirs(selected_beams_folder)
    except:
        print(f"Folder {selected_beams_folder} exists!")



    cir_cache = CIR_cache(PATH, FRAME_NUMBER, frames_per_data_frame=frames_per_data_frame)
    cir_cache.get_all_rewards()
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
    for i in range(ITER_NUMBER_CIR):
        oracle.append(max(cir_cache.all_rewards[:, i]))
    avarage_oracle = np.cumsum(oracle) / (np.arange(ITER_NUMBER_CIR) + 1)
    pickle.dump(oracle, open(
        f"{figures_path}/oracle_arms{int(ARMS_NUMBER_CIR)}.pickle", 'wb'))

    pickle.dump(np.array([cir_cache.max_reward]), open(
        f"{figures_path}/max_reward.pickle",
        'wb'))

    for SSB_period in SSB_periods:
        for n_b in NUMBERs_OF_CONS_SSB:
            calc(SSB_period,n_b)

