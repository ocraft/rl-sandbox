import math
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from rlbox.algorithm.policy_iteration import policy_iteration
from rlbox.env.car_rental import CarRentalEnv
from .sink import Store
from .testbed import Testbed


class CarRentalTestbed:

    summary = r'.dump/car_rental.h5'

    def __init__(self,
                 max_move, max_cars, expct, s0,
                 gamma=0.9, epsilon=1.0,
                 use_cache=True,
                 modified=False):
        self.env = CarRentalEnv(max_move=max_move, max_cars=max_cars,
                                expct=expct, s0=s0,
                                modified=modified)
        self.use_cache = use_cache
        self.gamma = gamma
        self.epsilon = epsilon

    def run(self):
        store = Store(self.summary, self._key)

        self.env.P = (store['mdp_p'][:]
                      if self.use_cache and 'mdp_p' in store
                      else None)
        self.env.R = (store['mdp_r'][:]
                      if self.use_cache and 'mdp_r' in store
                      else None)

        if self.env.P is None or self.env.R is None:
            start = time.perf_counter()
            self.env.prepare_model()
            print('Model TIME: ' + str(time.perf_counter() - start) + ' [s]')
            store.replace('mdp_p', self.env.P)
            store.replace('mdp_r', self.env.R)

        policies = (store['policy'][:]
                    if self.use_cache and 'policy' in store
                    else None)
        V = (store['value'][:]
             if self.use_cache and 'value' in store
             else None)

        if policies is None or V is None:
            start = time.perf_counter()
            policies, V = policy_iteration(
                mdp=self.env,
                policy=np.full(self.env.S.shape, 5, dtype=np.int32),
                gamma=self.gamma, epsilon=self.epsilon)
            print('Policy TIME: ' + str(time.perf_counter() - start) + ' [s]')
            store.replace('policy', policies)
            store.replace('value', V)

        store.close()

    def _key(self, value):
        params = {
            'max_cars': self.env.obs_spc['high'],
            'max_move': self.env.act_spc['high'],
            'req_l1': self.env.expct[0],
            'req_l2': self.env.expct[1],
            'ret_l1': self.env.expct[2],
            'ret_l2': self.env.expct[3],
            'modified': self.env.modified
        }

        return Testbed.key_for(value, **params)

    def plot(self):
        store = Store(self.summary, self._key)

        policies = store['policy'][:]
        V = store['value'][:]

        max_cars = self.env.obs_spc['high']
        max_move = self.env.act_spc['high']
        plt.style.use('seaborn')
        cols = 3
        fig, axes = plt.subplots(math.ceil(len(policies) / cols), cols,
                                 sharex='col', sharey='row',
                                 constrained_layout=True)

        for i, policy in enumerate(policies):
            row = (i // cols)
            col = i % cols
            ax = axes[row, col]
            im = ax.imshow(
                policy.reshape(max_cars + 1, max_cars + 1) - max_move,
                cmap="YlGnBu",
                origin='lower', interpolation='none',
                vmin=-max_move, vmax=max_move)
            ax.set_yticks(np.arange(max_cars, -2, -2))
            ax.set_xticks(np.arange(max_cars, -2, -2))

        fig.colorbar(im, ax=axes[:, -1], shrink=0.8)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        X = np.arange(0, max_cars + 1)
        Y = np.arange(0, max_cars + 1)
        X, Y = np.meshgrid(X, Y)
        ax.plot_surface(X, Y, V.reshape(max_cars + 1, max_cars + 1))

        plt.show()
        store.close()
