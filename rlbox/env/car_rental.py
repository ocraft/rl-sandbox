import functools
import itertools
from math import exp, factorial

import numpy as np

from .mdp import Mdp

DELTA = 0.001


@functools.lru_cache(maxsize=256)
def poisson(n, lam):
    return exp(-lam) * pow(lam, n) / factorial(n)


def max_poisson(lam):
    p = 1.0
    n = min(lam, 0)
    while p - DELTA > 0.0:
        p = poisson(n, lam)
        n += 1
    return n


class CarRentalEnv(Mdp):

    MOVE_REWARD = -2
    RENT_REWARD = 10
    PARK_REWARD = -4

    def __init__(self, max_move, max_cars, expct, s0, modified=False):
        Mdp.__init__(self, s0)
        self.act_spc = {'low': -max_move, 'high': max_move, 'shape': (1,)}
        self.obs_spc = {'low': 0, 'high': max_cars, 'shape': (2,)}
        self.expct = expct
        self._expct_max = [
            max_poisson(self.expct[0]) + 1,
            max_poisson(self.expct[1]) + 1,
            max_poisson(self.expct[2]) + 1,
            max_poisson(self.expct[3]) + 1
        ]
        self._state = s0
        self.A = np.arange(start=-max_move, stop=max_move+1)
        self.S = np.arange((max_cars + 1) * (max_cars + 1))
        self.modified = modified

    def step(self, action):
        act_rew = self._on_action(self._state, action)
        rent_rew = self._on_rent(
            self._state,
            np.random.poisson(lam=self.expct[0]),
            np.random.poisson(lam=self.expct[2]))
        ret_rew = self._on_return(
            self._state,
            np.random.poisson(lam=self.expct[1]),
            np.random.poisson(lam=self.expct[3]))

        return self._state, act_rew + rent_rew + ret_rew

    def _on_action(self, state, action):
        real_action = 0
        if action > 0:
            real_action = min(state[0], action)
        if action < 0:
            real_action = -min(state[1], abs(action))
        state[0] -= real_action
        state[1] += real_action

        if self.modified:
            rew = 0
            rew += abs(action - 1 if action > 0 else action) * self.MOVE_REWARD
            park_lmt = self.obs_spc['high'] / 2
            rew += self.PARK_REWARD * (
                (1 if state[0] > park_lmt else 0) +
                (1 if state[1] > park_lmt else 0))
            return rew
        return abs(action) * self.MOVE_REWARD

    def _on_rent(self, state, rent_loc_01, rent_loc_02):
        rent_loc01 = min(rent_loc_01, state[0])
        rent_loc02 = min(rent_loc_02, state[1])

        state[0] -= rent_loc01
        state[1] -= rent_loc02

        return (rent_loc01 + rent_loc02) * self.RENT_REWARD

    def _on_return(self, state, ret_loc_01, ret_loc_02):
        state[0] = min(state[0] + ret_loc_01, self.obs_spc['high'])
        state[1] = min(state[1] + ret_loc_02, self.obs_spc['high'])

        return 0

    def prepare_model(self):
        ns = len(self.S)
        na = len(self.A)

        # transition matrix
        t = np.zeros((ns, ns, na), dtype=np.float64)

        # reward matrix
        r = np.zeros(t.shape, dtype=np.float64)

        # probabilities of each rent/return situation
        p = np.zeros((
            self._expct_max[0],
            self._expct_max[1],
            self._expct_max[2],
            self._expct_max[3]))

        req_space = list(itertools.product(
            range(0, self._expct_max[0]),
            range(0, self._expct_max[1])))
        ret_space = list(itertools.product(
            range(0, self._expct_max[2]),
            range(0, self._expct_max[3])))

        max_cars = self.obs_spc['high']
        state_space = list(itertools.product(
            range(0, max_cars + 1),
            range(0, max_cars + 1)))

        state_idx = {s: i for i, s in enumerate(state_space)}

        for req_l1, req_l2 in req_space:
            for ret_l1, ret_l2 in ret_space:
                req_prob = (poisson(req_l1, self.expct[0]) *
                            poisson(req_l2, self.expct[1]))

                ret_prob = (poisson(ret_l1, self.expct[2]) *
                            poisson(ret_l2, self.expct[3]))
                p[req_l1, req_l2, ret_l1, ret_l2] = req_prob * ret_prob

        for i, (car_l1, car_l2) in enumerate(state_space):
            state = [car_l1, car_l2]
            for a, action in enumerate(self.A):
                state_n1 = state.copy()
                act_rew = self._on_action(state_n1, action)
                for req_l1, req_l2 in req_space:
                    state_n2 = state_n1.copy()
                    rent_rew = self._on_rent(state_n2, req_l1, req_l2)
                    for ret_l1, ret_l2 in ret_space:
                        state_n3 = state_n2.copy()
                        self._on_return(state_n3, ret_l1, ret_l2)

                        next_i = state_idx[(state_n3[0], state_n3[1])]
                        rew = (act_rew + rent_rew)
                        old_p = t[i, next_i, a]
                        _p = p[req_l1, req_l2, ret_l1, ret_l2]
                        new_p = old_p + _p

                        if new_p:
                            r[i, next_i, a] = (
                                ((r[i, next_i, a] * old_p) + rew * _p) / new_p)

                        t[i, next_i, a] = new_p

        self.P = t
        self.R = r
