

import pickle
import trimesh
import matplotlib.pyplot as plt

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

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory



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
        if self.iter_count < np.shape(ALL_RAWORD)[1]:
            #self.state = cir_cache.choose_context_number(self.context_set, self.iter_count)
            self.state = self.iter_count
        else:
            self.state = self.reset()

        reward = ALL_RAWORD[action, self.iter_count]
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




if __name__ == '__main__':
    # PLOTTING
    PLOT_ALL_REWARDS = True
    PLOT_CONTEXT = True
    PLOT_REWARDS_DESTRIBUTION = False

    # DATASET FROM ENGINE
    voxel_size = 0.5
    grid_step = 0.1
    P_TX = 1
    carrier_frequency = 900e6

    frames_per_data_frame = 1 #10000
    FRAME_NUMBER = 38
    ITER_NUMBER_CIR = frames_per_data_frame * FRAME_NUMBER
    ITER_NUMBER_RANDOM = ITER_NUMBER_CIR

    SUBDIVISION = 0
    icosphere = trimesh.creation.icosphere(subdivisions=SUBDIVISION, radius=1.0, color=None)
    #beam_directions = np.array(icosphere.vertices)
    beam_directions = np.array([np.array(icosphere.vertices)[1], np.array(icosphere.vertices)[8]])

    ARMS_NUMBER_CIR = len(beam_directions)
    SUBDIVISION_2 = 1
    icosphere_context = trimesh.creation.icosphere(subdivisions=SUBDIVISION_2, radius=1.0, color=None)



    NUMBER_OF_ITERATIONS_TRAINING = ITER_NUMBER_CIR #250000

    #context_sets = [np.array(icosphere_context.vertices),np.array([[1, -1, 0], [1, 1, 0], [-1, -1, 0], [-1, 1, 0]]), np.array([[1, 1, 0]])]
    context_sets = [np.array(icosphere_context.vertices)]


    algorithm_names = ["DQL"]

    parameters = [[0.05]]
    ALL_RAWORD = np.zeros((ARMS_NUMBER_CIR, ITER_NUMBER_CIR))
    ALL_RAWORD[0] = np.random.randint(2, size=(1, ITER_NUMBER_CIR))
    ALL_RAWORD[1] = abs((ALL_RAWORD[0] - 1))
    for con_set in context_sets:

        env = MultipathChannel(ARMS_NUMBER_CIR, ITER_NUMBER_CIR, con_set, starting_point = 0)
        #env_test = MultipathChannel( ARMS_NUMBER_CIR, len(con_set), con_set, starting_point = NUMBER_OF_ITERATIONS_TRAINING)
        env_test = MultipathChannel(ARMS_NUMBER_CIR, ITER_NUMBER_CIR, con_set,
                                    starting_point=0)


        for alg_name, pars in zip(algorithm_names, parameters):

            for p in pars:

                if alg_name == "DQL":

                    states = env.observation_space.shape
                    actions = env.action_space.n

                    model = build_model(states, actions)

                    dqn = build_agent(model, actions)

                    dqn.compile(Adam(lr=1e-3), metrics=['accuracy'])

                    number_of_cycles = 1000

                    dqn.fit(env, nb_steps=NUMBER_OF_ITERATIONS_TRAINING*number_of_cycles, visualize=False, verbose=1)

                    rewards_training = env.record_rewards
                    actions_training = env.record_actions
                    states_training = env.record_states

                    cum_rewards_training = np.cumsum(rewards_training) / (np.arange(len(rewards_training)) + 1)

                    plt.plot(cum_rewards_training)
                    plt.show()

                    number_of_steps_per_ep = ITER_NUMBER_CIR # ITER_NUMBER_CIR-NUMBER_OF_ITERATIONS_TRAINING
                    results = dqn.test(env_test, nb_episodes=1, nb_max_episode_steps = number_of_steps_per_ep, visualize=False)
                    rewards_test = env_test.record_rewards
                    actions_test = env_test.record_actions
                    states_test = env_test.record_states

                    all_rewards = rewards_test
                    all_actions = actions_test
                    all_states = states_test


                    # all_rewards = np.concatenate((rewards_training,rewards_test), axis = 0)
                    # all_actions = np.concatenate((actions_training,actions_test), axis = 0)
                    # all_states = np.concatenate((states_training, states_test), axis=0)

                    cumulative_average = np.cumsum(all_rewards) / (np.arange(len(all_rewards)) + 1)








