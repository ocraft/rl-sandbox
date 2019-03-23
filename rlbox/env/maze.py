import itertools

import numpy as np

from rlbox.core import Environment, Spec, Space

MAZE0 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 0, 3],
    [1, 1, 0, 1, 1, 1, 1, 0, 1],
    [2, 1, 0, 1, 1, 1, 1, 0, 1],
    [1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1]
])

MAZE1 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 1, 1, 1, 1, 1]
])

MAZE2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 1, 1, 1, 1, 1]
])


class Maze(Environment):
    WIDTH = 9
    HEIGHT = 6

    START = 2
    STOP = 3
    WALL = 0
    OK = 1

    ACT_SPACE = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    STATE_SPACE = list(itertools.product(range(0, WIDTH), range(0, HEIGHT)))

    STATE_IDX = {s: i for i, s in enumerate(STATE_SPACE)}
    ACT_IDX = {a: i for i, a in enumerate(ACT_SPACE)}

    S0 = [0, 3]
    GOAL = [WIDTH - 1, HEIGHT - 1]

    def __init__(self, maze_type=0):
        Environment.__init__(
            self,
            act_spec=Spec([Space(shape=(1,), domain=(0, 3))]),
            obs_spec=Spec([
                Space(shape=(1,), domain=(0, self.WIDTH - 1)),
                Space(shape=(1,), domain=(0, self.HEIGHT - 1))
            ]))

        self.maze_type = maze_type
        if maze_type == 0:
            self._init_maze(MAZE0)
        if maze_type == 1:
            self._init_maze(MAZE1)
        if maze_type == 2:
            self._init_maze(MAZE2)

        self.s = self.S0.copy()
        self.obs = tuple(self.s)
        self.reward = 0
        self.rewards = []
        self.nstep = 0
        self.steps_cnt = 0

    def _init_maze(self, blueprint):
        self.MAZE = np.flip(blueprint, axis=0).T
        self.GOAL = self._get(self.STOP)
        self.S0 = self._get(self.START)

    def _get(self, field):
        return list(list(zip(*np.where(self.MAZE == field)))[0])

    def reset(self):
        if self.maze_type == 1 and self.steps_cnt > 1000:
            self.maze_type = 2
            self._init_maze(MAZE2)
        self.s = self.S0.copy()
        self.obs = tuple(self.s)
        self.reward = None
        self.nstep = 0

    def done(self):
        return self.s == self.GOAL

    def step(self, action):
        action = self.ACT_SPACE[action]
        next_s = self.s.copy()

        next_s[0] += action[0]
        next_s[1] += action[1]
        next_s[0] = self._bound(next_s[0], 0, self.WIDTH - 1)
        next_s[1] = self._bound(next_s[1], 0, self.HEIGHT - 1)
        if self.MAZE[next_s[0], next_s[1]] != self.WALL:
            self.s = next_s

        self.obs = tuple(self.s)
        if self.done():
            self.reward = 1
        else:
            self.reward = 0
        self.rewards.append(self.reward)
        self.nstep += 1
        self.steps_cnt += 1
        return self.s, self.reward

    @staticmethod
    def _bound(value, vmin, vmax):
        if value < vmin:
            value = vmin
        if value > vmax:
            value = vmax
        return value
