from abc import abstractmethod

from rlbox.core import AgentProgram
from rlbox.rand import Sampler as rnd


class EpsilonGreedy(AgentProgram):
    def __init__(self, act_spec, obs_spec, epsilon, bias=0.0):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self.epsilon = epsilon
        self._q = [bias] * act_spec[0].size()  # action-value table

    def __call__(self, obs, reward):
        if obs is not None:
            self.update_action_values(obs, reward)

        if self.epsilon > rnd.rand():
            action = self.explore()
        else:
            action = self.best_action()

        return action

    @abstractmethod
    def update_action_values(self, act, rew):
        pass

    def explore(self):
        return rnd.rint(self.act_spec[0].size())

    def best_action(self):
        return self._q.index(max(self._q))

    def qtable(self):
        return self._q


class SampleAverage(EpsilonGreedy):
    def __init__(self, act_spec, obs_spec, epsilon):
        EpsilonGreedy.__init__(self, act_spec, obs_spec, epsilon)
        self._n = [0] * act_spec[0].size()  # taken action count table

    def update_action_values(self, act, rew):
        self._n[act] += 1
        self._q[act] += (rew - self._q[act]) / self._n[act]  # sample-average


class WeightedAverage(EpsilonGreedy):
    def __init__(self, act_spec, obs_spec, epsilon, alpha, bias=0.0):
        EpsilonGreedy.__init__(self, act_spec, obs_spec, epsilon, bias)
        self.alpha = alpha

    def update_action_values(self, act, rew):
        self._q[act] += (rew - self._q[act]) * self.alpha  # weighted-average


class WeightedAverageNBias(WeightedAverage):
    def __init__(self, act_spec, obs_spec, epsilon, alpha, bias=0.0):
        WeightedAverage.__init__(self, act_spec, obs_spec,
                                 epsilon, alpha, bias)
        self._o = 0.0

    def update_action_values(self, act, rew):
        self._o = self._o + self.alpha * (1 - self._o)
        self._q[act] += (rew - self._q[act]) * (self.alpha / self._o)
