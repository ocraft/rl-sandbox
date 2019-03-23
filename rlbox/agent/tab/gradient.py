import numpy as np
import numba

from rlbox.core import AgentProgram


class GradientBandit(AgentProgram):
    def __init__(self, act_spec, obs_spec, alpha, baseline=True):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self._a = np.arange(act_spec[0].size())
        self.alpha = alpha
        self.baseline = baseline
        self._r_base = 0
        self._t = 0
        self._h = np.zeros(act_spec[0].size())
        self._p = self._softmax(self._h)

    def __call__(self, obs, reward):
        self._t += 1
        if obs is not None:
            a = obs
            if self.baseline:
                r = reward - self._r_base
                self._r_base += r / self._t
            else:
                r = reward

            self._update_pref(self._h, self._a, a, self.alpha, r, self._p)

            self._p = self._softmax(self._h)

        return np.random.choice(self._a, p=self._p, replace=False)

    @staticmethod
    @numba.njit
    def _update_pref(h, act_space, a, alpha, r, p):
        h[a] += alpha * r * (1 - p[a])
        oth_a = act_space != a
        h[oth_a] -= alpha * r * p[oth_a]

    @staticmethod
    @numba.njit
    def _softmax(arr):
        return np.exp(arr) / np.sum(np.exp(arr), axis=0)

    def policy(self):
        return self._p

    def rbase(self):
        return self._r_base
