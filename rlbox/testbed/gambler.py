import time

import matplotlib.pyplot as plt

from rlbox.algorithm.policy_iteration import value_iteration
from rlbox.env.gambler import Gambler
from .sink import Store
from .testbed import Testbed


class GamblerTestbed:

    summary = r'.dump/gambler.h5'

    def __init__(self, ph, stake, gamma=0.9, epsilon=1.0, use_cache=True):
        self.env = Gambler(ph=ph, stake=stake)
        self.use_cache = use_cache
        self.gamma = gamma
        self.epsilon = epsilon

    def run(self):
        store = Store(self.summary, self._key)

        start = time.perf_counter()
        self.env.prepare_model()
        print('Model TIME: ' + str(time.perf_counter() - start) + ' [s]')
        store.replace('mdp_p', self.env.P)
        store.replace('mdp_r', self.env.R)

        start = time.perf_counter()
        policy, V = value_iteration(
            mdp=self.env, gamma=self.gamma, epsilon=self.epsilon)
        print('Value TIME: ' + str(time.perf_counter() - start) + ' [s]')
        store.replace('policy', policy)
        store.replace('value', V)

        store.close()

    def _key(self, value):
        params = {
            'ph': self.env.ph,
            'stake': self.env.stake
        }

        return Testbed.key_for(value, **params)

    def plot(self):
        store = Store(self.summary, self._key)

        policy = store['policy'][:]
        V = store['value'][:]

        plt.figure()
        plt.plot(policy[1:-1], drawstyle='steps')

        plt.figure()
        plt.plot(V[:, 1:-1].T)

        plt.show()
        store.close()
