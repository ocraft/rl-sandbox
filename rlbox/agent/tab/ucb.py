import numpy as np

from rlbox.core import AgentProgram


class Ucb1(AgentProgram):
    def __init__(self, act_spec, obs_spec, c, bias=0.0):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self._c = c
        self._q = np.full(act_spec[0].size(), bias)  # action-value table
        self._n = np.zeros(act_spec[0].size(), dtype=np.int32)
        self._t = 0
        self._has_unexplored = True

    def __call__(self, obs, reward):
        self._t += 1
        if obs is not None:
            self.update_action_values(obs, reward)

        return self.best_action()

    def update_action_values(self, act, rew):
        self._n[act] += 1
        self._q[act] += (rew - self._q[act]) / self._n[act]  # sample-average

    def best_action(self):
        if self._has_unexplored:
            max_action = np.argmax(self._n == 0)
            if self._n[max_action] == 0:
                return max_action
            self._has_unexplored = False
        return np.argmax(
            self._q + self._c * np.sqrt(np.log(self._t) / self._n))

    def qtable(self):
        return self._q
