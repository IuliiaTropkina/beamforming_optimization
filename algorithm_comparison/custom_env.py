from gym import Env
from gym.spaces import Box, Discrete
import random

import numpy as np


class ShowerEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.lowest_bound = 0
        self.highest_bound = 100
        self.observation_space = Box(low=np.array([self.lowest_bound]), high=np.array([self.highest_bound]))
        self.state = np.array([38 + random.randint(-3, 3)], dtype=np.float64)
        self.shower_length = 20
        self.record_rewards = []

    def step(self, action):
        self.state[0] += action - 1
        self.shower_length -= 1

        # Calculating the reward
        if self.state[0] >= 37 and self.state[0] <= 39:
            reward = 1
        else:
            reward = 0

            # Checking if shower is done

        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Setting the placeholder for info
        info = {}

        # Returning the step information
        self.record_rewards.append(reward)
        return self.state, reward, done, info

    def reset(self):
        self.state = np.array([38 + random.randint(-3, 3)], dtype=np.float64)
        self.shower_length = 20
        return self.state


class MultipathChannel(Env):
    def __init__(self, number_of_actions, number_of_states, number_of_iterations):
        self.action_space = Discrete(number_of_actions)
        self.observation_space = Box(low=np.array([0, 1, 2, 3]), high=np.array(
            [number_of_states, number_of_states + 1, number_of_states + 2, number_of_states + 3]))
        self.iter_count = 0
        self.record_rewards = []
        self.record_actions = []
        self.record_states = []
        self.ALL_REWARD = np.zeros((number_of_actions, number_of_iterations))
        self.create_reward_table(3)

        self.state = np.array([0.0, 1.0, 2.0, 3.0])

    def create_reward_table(self, seed):
        # np.random.seed(seed)
        # self.ALL_REWARD[0] = np.random.randint(2, size=(1, len(self.ALL_REWARD[0])))

        self.ALL_REWARD[0, 0:int(np.shape(self.ALL_REWARD)[1] / 2)] = 1

        # self.ALL_REWARD[0] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0,
        #         0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0,
        #         1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1])

        # self.ALL_REWARD[0] = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0])

        self.ALL_REWARD[1] = abs((self.ALL_REWARD[0] - 1))

    def step(self, action):
        reward = self.ALL_REWARD[action, self.iter_count]
        self.record_rewards.append(reward)
        self.record_actions.append(action)
        self.record_states.append(self.state[0])

        self.iter_count += 1
        self.state[0] = self.iter_count
        self.state[1] = self.iter_count + 1
        self.state[2] = self.iter_count + 2
        self.state[3] = self.iter_count + 3
        if self.iter_count >= self.ALL_REWARD.shape[1]:
            done = True
        else:
            done = False

        return self.state.copy(), reward, done, {}

    def reset(self, **_):
        self.iter_count = 0
        self.record_rewards = []
        self.record_actions = []
        self.record_states = []
        # self.state = cir_cache.choose_context_number(self.context_set, self.starting_point)
        self.state[0] = 0.0
        self.state[1] = 1.0
        self.state[2] = 2.0
        self.state[3] = 3.0

        return self.state



