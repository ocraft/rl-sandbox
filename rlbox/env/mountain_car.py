import math

import numpy as np

from rlbox.core import Environment, Spec, Space


class MountainCar(Environment):
    BOUND_MIN_P = -1.2
    BOUND_MAX_P = 0.5

    BOUND_MIN_V = -0.07
    BOUND_MAX_V = 0.07

    ACT_SPACE = [-1, 0, 1]
    ACT_IDX = {a: i for i, a in enumerate(ACT_SPACE)}

    def __init__(self):
        Environment.__init__(
            self,
            act_spec=Spec([Space(shape=(1,), domain=(-1, 1))]),
            obs_spec=Spec([
                Space(shape=(1,),
                      dtype=np.float32,
                      domain=(self.BOUND_MIN_P, self.BOUND_MAX_P)),
                Space(shape=(1,),
                      dtype=np.float32,
                      domain=(self.BOUND_MIN_V, self.BOUND_MAX_V))
            ]))

        self.p = np.random.uniform(-0.6, -0.4)
        self.v = 0.0
        self.obs = (self.p, self.v)
        self.reward = None
        self.nstep = 0

    def reset(self):
        self.p = np.random.uniform(-0.6, -0.4)
        self.v = 0.0
        self.obs = (self.p, self.v)
        self.reward = None
        self.nstep = 0

    def done(self):
        return self.p == self.BOUND_MAX_P

    def step(self, action):
        action = self.ACT_SPACE[action]

        self.v = self._bound_v(
            self.v + 0.001 * action - 0.0025 * math.cos(3 * self.p))
        self.p = self._bound_p(self.p + self.v)
        if self.p == self.BOUND_MIN_P:
            self.v = 0.0

        self.obs = (self.p, self.v)
        self.reward = -1
        self.nstep += 1

        return self.obs, self.reward

    def _bound_p(self, value):
        if value > self.BOUND_MAX_P:
            return self.BOUND_MAX_P
        if value < self.BOUND_MIN_P:
            return self.BOUND_MIN_P
        return value

    def _bound_v(self, value):
        if value > self.BOUND_MAX_V:
            return self.BOUND_MAX_V
        if value < self.BOUND_MIN_V:
            return self.BOUND_MIN_V
        return value
