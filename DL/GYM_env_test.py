import numpy as np
from gym import Env
from gym.spaces import Box, Discrete
import random


class CustomEnv(Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=np.array([0]), high=np.array([100]))
        self.state = 38 + random.randint(-3,3)
        self.shower_length = 60

    def step(self, action):
        self.state += action - 1
        self.shower_length -= 1

        # Calculating the reward
        if self.state >= 37 and self.state <= 39:
            reward = 1
        else:
            reward = -1

            # Checking if shower is done
        if self.shower_length <= 0:
            done = True
        else:
            done = False

        # Setting the placeholder for info
        info = {}

        # Returning the step information
        return self.state, reward, done, info

    def reset(self):
        self.state = 38 + random.randint(-3, 3)
        self.shower_length = 60
        return self.state



env = CustomEnv()

episodes = 20  # 20 shower episodes
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))

