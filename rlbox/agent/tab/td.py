import numpy as np

from rlbox.core import AgentProgram
from rlbox.core.space import Spec
from rlbox.rand import Sampler as rnd
from rlbox.rand import cache_gen


class OnPolicySarsa(AgentProgram):
    def __init__(self, act_spec: Spec, obs_spec: Spec, alpha=0.5, epsilon=0.1,
                 gamma=1.0):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.q = np.zeros(shape=(obs_spec.size(), act_spec.size()))
        self.sample = cache_gen(
            lambda: np.random.randint(0, act_spec.size(), 1000))

        self.s = None
        self.a = None

    def reset(self):
        self.s = None
        self.a = None

    def __call__(self, obs, reward):
        if self.epsilon > rnd.rand():
            action = next(self.sample)
        else:
            action = np.argmax(self.q[obs, :])

        if self.s is not None and self.a is not None:
            self.q[self.s, self.a] += self.alpha * (
                    reward +
                    self.gamma * self.q[obs, action] -
                    self.q[self.s, self.a])

        self.s = obs
        self.a = action

        return action


def e_greedy_policy(pi, q, epsilon):
    for s, _ in enumerate(q):
        a_star = np.argmax(q[s, :])
        a_space = len(q[s, :])
        e_soft = epsilon / a_space
        pi[s, a_star] = 1 - epsilon + e_soft
        pi[s, np.arange(a_space) != a_star] = e_soft
    return pi


class OnPolicyNStepSarsa(AgentProgram):
    def __init__(self, act_spec: Spec, obs_spec: Spec, n=2,
                 alpha=0.5, epsilon=0.1, gamma=1.0):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.n = n
        self.q = np.zeros(shape=(obs_spec.size(), act_spec.size()),
                          dtype=np.float64)
        self.actions = np.arange(act_spec.size())
        self.pi = e_greedy_policy(
            np.zeros(shape=(obs_spec.size(), act_spec.size()),
                     dtype=np.float64),
            self.q, self.epsilon)
        self._t = 0
        self._T = float('inf')
        self._S = []
        self._A = []
        self._R = []
        self._tau = 0

    def reset(self):
        self._t = 0
        self._T = float('inf')
        self._S = []
        self._A = []
        self._R = []
        self._tau = 0

    def __call__(self, obs, reward, done=False):
        act = None
        if self._t <= self._T:
            self._S.append(obs)
            if self._t > 0:
                self._R.append(reward)
            if done:
                self._T = self._t
            else:
                act = np.random.choice(a=self.actions,
                                       p=self.pi[obs, :])
                self._A.append(act)

        self._tau = self._t - self.n
        if self._tau >= 0:
            step = self._tau + self.n
            G = 0.0
            for i in range(self._tau + 1, min(step, self._T) + 1):
                G += (self.gamma ** (i - self._tau - 1)) * self._R[i - 1]
            if step < self._T:
                G += ((self.gamma ** self.n) *
                      self.q[self._S[step], self._A[step]])

            S = self._S[self._tau]
            A = self._A[self._tau]
            self.q[S, A] += self.alpha * (G - self.q[S, A])

            a_star = np.argmax(self.q[S, :])
            a_space = self.act_spec.size()
            e_soft = self.epsilon / a_space
            self.pi[S, a_star] = 1.0 - self.epsilon + e_soft
            self.pi[S, self.actions != a_star] = e_soft

        self._t += 1
        if done and (self._tau < self._T - 1):
            self(obs, reward, done)
        return act
