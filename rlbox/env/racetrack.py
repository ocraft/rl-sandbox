import itertools
import random

import numpy as np

from rlbox.core import Environment, Spec, Space
from rlbox.rand import Sampler as rnd

TRACK = np.array([
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0]
], dtype=np.int32)


def path(start, end):
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


class RaceTrack(Environment):
    OUT = 0
    ON = 1
    START = 2
    END = 3

    P_FAIL = 0.1
    MAX_VEL = 5

    STATE_SPACE = list(itertools.product(
        range(0, TRACK.shape[0]),
        range(0, TRACK.shape[1]),
        range(0, MAX_VEL),
        range(0, MAX_VEL)))

    ACT_SPACE = list(itertools.product(range(-1, 2), range(-1, 2)))

    STATE_IDX = {s: i for i, s in enumerate(STATE_SPACE)}
    ACT_IDX = {a: i for i, a in enumerate(ACT_SPACE)}

    def __init__(self, steps=10000):
        Environment.__init__(
            self,
            act_spec=Spec([Space(shape=(2,), domain=(-1, 1))]),
            obs_spec=Spec([
                Space(shape=(1,), domain=(0, TRACK.shape[0] - 1)),
                Space(shape=(1,), domain=(0, TRACK.shape[1] - 1)),
                Space(shape=(1,), domain=(0, self.MAX_VEL - 1)),
                Space(shape=(1,), domain=(0, self.MAX_VEL - 1))
            ]))
        self.steps = steps
        self.start_loc = self._track_part(self.START)
        self.pos = list(random.choice(self.start_loc))
        self.vel = [0, 0]
        self._done = False
        self._step = 0
        self.states = [-1] * steps
        self.actions = [-1] * steps
        self.rewards = [0] * steps
        self.obs = tuple(self.pos + self.vel)

    @staticmethod
    def _track_part(track_type):
        return list(zip(*np.where(TRACK == track_type)))

    def loc(self, pos):
        if (pos[0] >= TRACK.shape[0]
                or pos[1] >= TRACK.shape[1]
                or pos[0] < 0
                or pos[1] < 0):
            return self.OUT
        return TRACK[pos[0], pos[1]]

    def done(self):
        return self._done or self._step >= self.steps

    def success(self):
        return self._done and self._step < self.steps

    def step(self, action):
        vx = action[0]
        vy = action[1]
        if self.P_FAIL > rnd.rand():
            vx, vy = 0, 0

        self.vel[0] += vx
        self.vel[1] += vy

        if self.vel[0] >= self.MAX_VEL:
            self.vel[0] = self.MAX_VEL - 1
        if self.vel[1] >= self.MAX_VEL:
            self.vel[1] = self.MAX_VEL - 1
        if self.vel[0] < 0:
            self.vel[0] = 0
        if self.vel[1] < 0:
            self.vel[1] = 0

        if self.vel == [0, 0]:
            self.vel = [1, 1]

        obs0 = self.obs
        start_pos = self.pos.copy()
        end_pos = self.pos.copy()

        end_pos[0] -= self.vel[0]
        end_pos[1] += self.vel[1]

        track_states = list(map(self.loc, path(start_pos, end_pos)))
        if self.END in track_states:
            self._done = True
        elif self.OUT in track_states:
            self.pos = list(random.choice(self.start_loc))
            self.vel = [0, 0]
        else:
            self.pos = end_pos

        self.obs = tuple(self.pos + self.vel)
        self.states[self._step] = self.STATE_IDX[obs0]
        self.actions[self._step] = self.ACT_IDX[tuple(action)]
        self.rewards[self._step] = -1
        self._step += 1
        return self.obs, -1

    POS_TO_RGB = {
        END: (1, 0, 0),
        ON: (139 / 255, 69 / 255, 19 / 255),
        OUT: (.5, 1, 0),
        START: (1, 1, 0)
    }

    def episode(self):
        st = np.full(self.steps, fill_value=-1, dtype=np.int32)
        act = np.full(self.steps, fill_value=-1, dtype=np.int32)
        rew = np.zeros(self.steps, dtype=np.int32)
        if self.success():
            np.copyto(dst=st[:self._step],
                      src=list(reversed(self.states[:self._step])))
            np.copyto(dst=act[:self._step],
                      src=list(reversed(self.actions[:self._step])))
            np.copyto(dst=rew[:self._step],
                      src=list(reversed(self.rewards[:self._step])))
        return st, act, rew

    def print(self):
        tr_rgb = [[self.POS_TO_RGB[s] for s in row] for row in TRACK]
        x, y = self.pos
        tr_rgb[x][y] = (0, 0, 0)

        return tr_rgb
