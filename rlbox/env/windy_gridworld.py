import itertools

import numpy as np

from rlbox.core import Environment, Spec, Space


class WindyGridWorld(Environment):
    WIDTH = 10
    HEIGHT = 7
    GOAL = [7, 3]
    WIND = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    ACT_SPACE = [(0, 1), (0, -1), (1, 0), (-1, 0), (-1, 1), (1, 1), (1, -1),
                 (-1, -1)]
    STATE_SPACE = list(itertools.product(range(0, WIDTH), range(0, HEIGHT)))

    STATE_IDX = {s: i for i, s in enumerate(STATE_SPACE)}
    ACT_IDX = {a: i for i, a in enumerate(ACT_SPACE)}

    S0 = [0, 3]

    def __init__(self, stochastic=False):
        Environment.__init__(
            self,
            act_spec=Spec([Space(shape=(1,), domain=(0, 7))]),
            obs_spec=Spec([
                Space(shape=(1,), domain=(0, self.WIDTH - 1)),
                Space(shape=(1,), domain=(0, self.HEIGHT - 1))
            ]))
        self.stochastic = stochastic
        self.s = self.S0.copy()
        self.obs = tuple(self.s)
        self.reward = None
        self.nstep = 0

    def done(self):
        return self.s == self.GOAL

    def reset(self):
        self.s = self.S0.copy()
        self.obs = tuple(self.s)
        self.reward = None
        self.nstep = 0

    def step(self, action):
        action = self.ACT_SPACE[action]
        wind = self.WIND[self.s[0]]
        if self.stochastic:
            wind = np.random.randint(wind - 1, wind + 2)
        self.s[1] += wind + action[1]
        self.s[0] += action[0]
        self.s[0] = self._bound(self.s[0], 0, self.WIDTH - 1)
        self.s[1] = self._bound(self.s[1], 0, self.HEIGHT - 1)
        self.nstep += 1
        self.reward = -1
        self.obs = tuple(self.s)
        return self.s, self.reward

    @staticmethod
    def _bound(value, vmin, vmax):
        if value < vmin:
            value = vmin
        if value > vmax:
            value = vmax
        return value
