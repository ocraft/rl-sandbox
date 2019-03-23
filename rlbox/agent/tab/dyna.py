import math

import numpy as np

from rlbox.core import AgentProgram
from rlbox.core.space import Spec
from rlbox.rand import Sampler as rnd
from rlbox.rand import cache_gen


class DynaQ(AgentProgram):
    def __init__(self, act_spec: Spec, obs_spec: Spec, alpha=0.5, epsilon=0.1,
                 gamma=1.0, n=5, kappa=0):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.kappa = kappa
        self.n = n
        self.q = np.zeros(shape=(obs_spec.size(), act_spec.size()))
        self.sample = cache_gen(
            lambda: np.random.randint(0, act_spec.size(), 1000))

        if self.kappa:
            self.model = TimeModel(act_spec, obs_spec, kappa)
        else:
            self.model = Model(act_spec, obs_spec)

        self.s = None
        self.a = None

    def reset(self):
        self.s = None
        self.a = None

    def __call__(self, obs, reward):
        action = self.action_for(obs)

        if self.s is not None and self.a is not None:
            self.update_q(self.s, self.a, obs, reward)
            self.model.feed(self.s, self.a, obs, reward)

        self.s = obs
        self.a = action

        for _ in range(0, self.n):
            sample = self.model.sample()
            if sample:
                self.update_q(*sample)

        return action

    def action_for(self, obs):
        if self.epsilon > rnd.rand():
            action = next(self.sample)
        else:
            action = np.random.choice(
                np.flatnonzero(self.q[obs, :] == self.q[obs, :].max()))
        return action

    def update_q(self, s0, a0, s1, r1):
        self.q[s0, a0] += self.alpha * (
                r1 +
                self.gamma * np.max(self.q[s1, :]) -
                self.q[s0, a0])


class Model:
    def __init__(self, act_spec, obs_spec):
        self.model_s = np.full(shape=(obs_spec.size(), act_spec.size()),
                               fill_value=-1,
                               dtype=np.int32)
        self.model_r = np.zeros(shape=(obs_spec.size(), act_spec.size()),
                                dtype=np.int32)
        self.act_spec = act_spec
        self.obs_spec = obs_spec

    def feed(self, s0, a0, s1, r1):
        self.model_s[s0, a0] = s1
        self.model_r[s0, a0] = r1

    def sample(self):
        ks, ka = np.asarray(self.model_s > -1).nonzero()
        if ks.size == 0:
            return None
        i = np.random.randint(len(ks))
        s0, a0 = ks[i], ka[i]

        s1 = self.model_s[s0, a0]
        r1 = self.model_r[s0, a0]

        return s0, a0, s1, r1


class TimeModel(Model):
    def __init__(self, act_spec, obs_spec, kappa):
        Model.__init__(self, act_spec=act_spec, obs_spec=obs_spec)
        self.kappa = kappa
        self.tau = np.ones(shape=(obs_spec.size(), act_spec.size()),
                           dtype=np.int32)
        self.t = 0

    def feed(self, s0, a0, s1, r1):
        self.t += 1
        self.tau[s0, a0] = self.t
        self.model_s[s0, a0] = s1
        self.model_r[s0, a0] = r1

    def sample(self):
        ks = np.asarray(~np.all(self.model_s == -1, axis=1)).nonzero()[0]
        if ks.size == 0:
            return None
        s0 = np.random.choice(ks)
        a0 = np.random.randint(self.act_spec.size())

        if self.model_s[s0, a0] != -1:
            s1 = self.model_s[s0, a0]
            r1 = self.model_r[s0, a0]
        else:
            s1 = s0
            r1 = 0

        r1 += self.kappa * math.sqrt(self.t - self.tau[s0, a0])

        return s0, a0, s1, r1


class DynaQv2(DynaQ):
    def __init__(self, act_spec: Spec, obs_spec: Spec, alpha=0.5, epsilon=0.1,
                 gamma=1.0, n=5, kappa=0):
        DynaQ.__init__(self, act_spec, obs_spec, alpha, epsilon, gamma, n,
                       kappa)
        self.model = Model(act_spec, obs_spec)
        self.t = 0
        self.tau = np.ones(shape=(obs_spec.size(), act_spec.size()),
                           dtype=np.int32)

    def action_for(self, obs):
        self.t += 1
        if self.epsilon > rnd.rand():
            action = next(self.sample)
        else:
            q = self.q[obs, :] + self.kappa * np.sqrt(
                self.t - self.tau[obs, :])
            action = np.random.choice(np.flatnonzero(q == q.max()))
        self.tau[obs, action] = self.t
        return action
