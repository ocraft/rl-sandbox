import numba
import numpy as np

import rlbox.core as core
from rlbox.rand import Sampler as rnd


class Bandit:
    def __init__(self, arms, mean=0.0):
        self.qstar_means = np.random.normal(size=arms, loc=mean)
        self.best_action = np.argmax(self.qstar_means)

    def pull(self, arm):
        return rnd.randn() + self.qstar_means[arm]


class NonstationaryBandit:
    def __init__(self, arms, mean=0.0):
        self.arms = arms
        self.qstar_means = np.full(arms, np.random.normal(size=1, loc=mean))
        self.best_action = 0

    def pull(self, arm):
        self.qstar_means = self._walk(self.qstar_means)
        self.best_action = self._best(self.qstar_means)
        reward = rnd.randn() + self.qstar_means[arm]
        return reward

    @staticmethod
    @numba.njit
    def _walk(arr):
        return arr + np.random.normal(0.0, 0.01, len(arr))

    @staticmethod
    @numba.njit
    def _best(arr):
        return np.argmax(arr)


class NArmedBanditEnv(core.Environment):
    def __init__(self, steps, arms, stationary=True, mean=0.0):
        core.Environment.__init__(self,
                                  core.Spec([core.Space(shape=(arms,))]),
                                  core.Spec([core.Space(shape=(arms,))]))
        self.bandit = (Bandit(arms, mean)
                       if stationary
                       else NonstationaryBandit(arms, mean))
        self.steps = steps
        self._step = 0
        self.all_rewards = [0.0] * steps
        self.optimal_actions = [0] * steps
        self.last_action = None
        self.reward = 0

    def step(self, action):
        reward = self.bandit.pull(action)
        self.all_rewards[self._step] = reward
        self.optimal_actions[self._step] = action == self.bandit.best_action
        self._step += 1
        self.last_action = action
        self.reward = reward
        return action, reward

    def done(self):
        return self._step >= self.steps
