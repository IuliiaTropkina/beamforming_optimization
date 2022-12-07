
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict
from abc import ABC

import numpy as np
from numpy.random import normal

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy.random import uniform, normal
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List
from tqdm import trange
from uuid import uuid1

import numpy as np
from math import log, sqrt
from typing import List
from abc import ABC, abstractmethod

from abc import ABC, abstractmethod
from numpy.random import uniform
from typing import List, Dict

# from .multi_armed_bandit import MultiArmedBandit

class Algorithm(ABC):
    _n_arms: int

    def __init__(self, n_arms: int):
        self._id = uuid1()
        self._n_arms = n_arms

    def get_id(self):
        return self._id

    @abstractmethod
    def reset_agent(self):
        pass

    @abstractmethod
    def update_estimates(self, action: int, reward: int) -> None:
        pass

    @abstractmethod
    def select_action(self) -> int:
        pass

    @abstractmethod
    def plot_estimates(self):
        pass



class MultiArmedBandit(ABC):
    _n_arms: int
    _best_action: float

    def __init__(self, n_arms: int):
        self._n_arms = n_arms

    def get_n_arms(self):
        return self._n_arms

    @abstractmethod
    def plot_arms(self):
        pass

    @abstractmethod
    def do_action(self, action: int):
        pass

    @abstractmethod
    def best_action_mean(self):
        pass

    @abstractmethod
    def get_best_action(self):
        pass

    @abstractmethod
    def action_mean(self, action: int):
        pass

class DynamicMultiArmedBandit(MultiArmedBandit, ABC):
    _prob_of_change: float
    _action_value_trace: Dict
    _fixed_actions: List = []
    _type_change: str

    def __init__(self, n_arms: int, prob_of_change: float = 0.001, fixed_action_prob: float = None,
                 type_change: str = 'abrupt'):
        super().__init__(n_arms)
        self._prob_of_change = prob_of_change
        self._type_change = type_change

        if fixed_action_prob != None:
            for i in range(n_arms):
                if uniform(0, 1) < fixed_action_prob:
                    self._fixed_actions.append(i)

    @abstractmethod
    def change_action_prob(self, step: int):
        pass

class GaussianAlgo(Algorithm, ABC):
    _mu: List[float]
    _std_dev: List[float]
    _n_action_taken: List[int]
    _mean_trace: Dict

    def __init__(self, n_arms: int, decay_rate: float = 0.99):
        super().__init__(n_arms=n_arms)
        self._mu = np.ones(n_arms) / 2
        self._std_dev = np.ones(n_arms)
        self._decay_rate = decay_rate
        self._n_action_taken = np.zeros(n_arms)
        self._mean_trace = {action: [1 / 2] for action in range(n_arms)}

    def reset_agent(self):
        self._mu = np.ones(self._n_arms) / 2
        self._std_dev = np.ones(self._n_arms)
        self._n_action_taken = np.zeros(self._n_arms)
        self._mean_trace = {action: [1 / 2] for action in range(self._n_arms)}

    def update_estimates(self, action: int, reward: int) -> None:
        self._n_action_taken[action] += 1
        self._std_dev[action] *= self._decay_rate
        n = self._n_action_taken[action]
        if n == 0:
            self._mu[action] = reward
            return
        self._mu[action] += (1 / n) * (reward - self._mu[action])
        for a in range(self._n_arms):
            self._mean_trace[a].append(self._mu[a])

    def plot_estimates(self, render: bool = True):
        fig = plt.figure()
        for a in range(self._n_arms):
            _ = plt.plot(self._mean_trace[a], label="Action: " + str(a))
        fig.suptitle("Action's estimates")
        fig.legend()
        if render:
            fig.show()


class GaussianThompsonSampling(GaussianAlgo):

    def __init__(self, n_arms: int, decay_rate: float = 0.99):
        super().__init__(n_arms, decay_rate)

    def __repr__(self):
        return "Thompson Samplig gaussian, decay rate: " + str(self._decay_rate)

    def select_action(self) -> int:
        samples = [normal(loc=self._mu[a], scale=self._std_dev[a]) for a in range(self._n_arms)]
        return np.argmax(samples)


class GaussianBandit(MultiArmedBandit):
    _mean: List[float]
    _std_dev: List[float]

    def __init__(self, n_arms: int, mean: List[float] = None, std_dev: List[float] = None):

        super().__init__(n_arms)

        if mean == None:
            self._mean = [uniform(0, 1) for _ in range(n_arms)]
        elif mean != None and n_arms == len(mean):
            self._mean = mean
        elif mean != None and n_arms != len(mean):
            raise Exception(
                "Length of mean vector must be the same of number of arms")

        if std_dev == None:
            self._std_dev = [uniform(0.5, 0.9) for _ in range(n_arms)]
        elif isinstance(std_dev, float) or isinstance(std_dev, int):
            self._std_dev = [std_dev for _ in range(n_arms)]
        elif std_dev != None and n_arms == len(std_dev):
            self._std_dev = std_dev
        elif std_dev != None and n_arms != len(std_dev):
            raise Exception(
                "Length of standard deviation vector must be the same of number of arms")

        self._best_action = np.argmax(self._mean)

    def __repr__(self):
        return "Gaussian Multi-armed bandit\n" + \
               "Mean = " + str(self._mean) + \
               "\nStandard Deviation = " + str(self._std_dev)

    def plot_arms(self, render: bool = True):
        plt.figure()
        for a in range(self._n_arms):
            x = np.linspace(self._mean[a] - 3 * self._std_dev[a], self._mean[a] + 3 * self._std_dev[a])
            plt.plot(x,
                     stats.norm.pdf(x, self._mean[a], self._std_dev[a]),
                     label="Action: " + str(a) + ", Mean: " + str(self._mean[a]) + ", std_dev: " + str(
                         self._std_dev[a]))
        plt.suptitle("Bandit's arms values")
        plt.legend()
        if render:
            plt.show()

    def do_action(self, action: int):
        return normal(loc=self._mean[action], scale=self._std_dev[action])

    def best_action_mean(self):
        return self._mean[self._best_action]

    def get_best_action(self):
        return self._best_action

    def action_mean(self, action: int):
        return self._mean[action]


class Session():
    _regret: Dict
    _action_selection_trace: Dict
    _action_selection: Dict
    _real_reward_trace: Dict
    _real_reward: Dict
    _env: MultiArmedBandit
    _agents: List[Algorithm]

    def __init__(self, env: MultiArmedBandit, agent: List[Algorithm]):
        self._env = env
        if isinstance(agent, List):
            self._agents = agent
        else:
            self._agents = [agent]

    def run(self, n_step: int, n_test: int = 1, use_replay: bool = False) -> None:

        # Save statistics
        self._regrets = {agent: np.zeros(n_step) for agent in self._agents}

        self._real_reward_trace = {agent: np.zeros(n_step) for agent in self._agents}
        self._real_reward_trace.update({"Oracle": np.zeros(n_step)})

        self._action_selection_trace = {agent: {a: np.zeros(n_step) for a in range(self._env.get_n_arms())} for agent in
                                        self._agents}

        for test in trange(n_test):
            self._real_reward = {agent: 0 for agent in self._agents}
            self._real_reward.update({"Oracle": 0})
            self._action_selection = {agent: {a: 0 for a in range(self._env.get_n_arms())} for agent in self._agents}

            for step in range(n_step):
                # Oracle
                self._real_reward["Oracle"] += self._env.action_mean(self._env.get_best_action())
                self._real_reward_trace["Oracle"][step] += (1 / (test + 1)) * (
                            self._real_reward["Oracle"] - self._real_reward_trace["Oracle"][step])

                for agent in self._agents:
                    action = agent.select_action()
                    reward = self._env.do_action(action)
                    agent.update_estimates(action, reward)
                    self._update_statistic(test=test, step=step, id_agent=agent, action=action)

                if isinstance(self._env, DynamicMultiArmedBandit):
                    self._env.change_action_prob(step=step)

            # Reset env and agent to start condition, then the changes will be those stored in the replay saved inside the env
            if (test < n_test - 1) and use_replay:
                self._env.reset_to_start()
                for agent in self._agents:
                    agent.reset_agent()

    def _update_statistic(self, test, step, id_agent, action):
        reward = self._env.action_mean(action)

        self._regrets[id_agent][step] += (1 / (test + 1)) * (
                    self._env.best_action_mean() - reward - self._regrets[id_agent][step])

        self._real_reward[id_agent] += reward
        self._real_reward_trace[id_agent][step] += (1 / (test + 1)) * (
                    self._real_reward[id_agent] - self._real_reward_trace[id_agent][step])

        for a in range(self._env.get_n_arms()):
            if a == action:
                self._action_selection[id_agent][action] += 1
                self._action_selection_trace[id_agent][action][step] += (1 / (test + 1)) * (
                            self._action_selection[id_agent][action] - self._action_selection_trace[id_agent][action][
                        step])
            else:
                self._action_selection_trace[id_agent][a][step] += (1 / (test + 1)) * (
                            self._action_selection[id_agent][a] - self._action_selection_trace[id_agent][a][step])

    def plot_regret(self, render: bool = True):
        plt.figure()
        for agent in self._agents:
            plt.plot(self._regrets[agent], label=agent)
        plt.suptitle("Regret")
        plt.legend()
        if render:
            plt.show()

    def plot_action_selection_trace(self, render: bool = True):
        if len(self._agents) > 1:
            fig, axs = plt.subplots(len(self._agents), sharey=True)
            for i, agent in enumerate(self._agents):
                for action in range(self._env.get_n_arms()):
                    axs[i].plot(self._action_selection_trace[agent][action], label="Action: " + str(action))
                    axs[i].set_title("Action selection: " + str(agent))
                    axs[i].legend()
                fig.suptitle("Action selection")
        else:
            fig = plt.figure()
            agent = self._agents[0]
            for action in range(self._env.get_n_arms()):
                plt.plot(self._action_selection_trace[agent][action], label="Action: " + str(action))
                plt.legend()
            plt.suptitle("Action selection")

        if render:
            plt.show()

    def plot_real_reward_trace(self, render: bool = True):
        plt.figure()
        plt.plot(self._real_reward_trace["Oracle"], label="Oracle")
        for agent in self._agents:
            plt.plot(self._real_reward_trace[agent], label=agent)
        plt.suptitle("Real rewards sum")
        plt.legend()
        if render:
            plt.show()

    def plot_all(self, render: bool = True):
        self.plot_regret(render=False)
        self.plot_action_selection_trace(render=False)
        self.plot_real_reward_trace(render=False)
        if render:
            plt.show()

    def get_reward_sum(self, agent: Algorithm):
        if agent == "Oracle":
            return self._real_reward_trace["Oracle"][-1]
        return self._real_reward_trace[agent][-1]
class GaussianGreedy(GaussianAlgo):

    def __init__(self, n_arms: int, decay_rate: float = 0.99):
        super().__init__(n_arms, decay_rate)

    def __repr__(self):
        return "Greedy gaussian, decay rate: " + str(self._decay_rate)

    def select_action(self) -> int:
        return np.argmax(self._mu)






class GaussianUCB(GaussianAlgo):
    _action_taken: int
    _action_selection: List
    _c: float

    def __init__(self, n_arms: int, c: float = 1, decay_rate: float = 0.99):
        super().__init__(n_arms=n_arms, decay_rate=decay_rate)
        self._action_taken = 0
        self._action_selection = [0 for _ in range(n_arms)]
        self._c = c

    def __repr__(self):
        return "UCB gaussian, decay rate: " + str(self._decay_rate)

    def select_action(self) -> int:
        self._action_taken += 1
        estimates = []
        for a in range(self._n_arms):
            if self._action_selection[a] == 0:
                estimates.append(float("inf"))
            else:
                estimates.append(self._mu[a] + self._c * sqrt(log(self._action_taken) / self._action_selection[a]))

        action = np.argmax(estimates)
        self._action_selection[action] += 1
        return action


def test_gaussian_algorithms():
    n_arms = 4
    env = GaussianBandit(n_arms, std_dev=0.7)

    greedy_agent = GaussianGreedy(n_arms)
    ts_agent = GaussianThompsonSampling(n_arms, decay_rate=0.90)
    ucb_agent = GaussianUCB(n_arms, decay_rate=0.9, c=1)

    session = Session(env, [greedy_agent, ts_agent, ucb_agent])
    session.run(3000)

    env.plot_arms()
    session.plot_regret()
    session.plot_action_selection_trace()
    session.plot_real_reward_trace()
    # session.get_reward_sum(ts_agent)
    plt.show()


test_gaussian_algorithms()