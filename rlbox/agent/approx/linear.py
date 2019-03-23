import numpy as np

from rlbox.core import AgentProgram
from rlbox.core.space import Spec
from rlbox.rand import Sampler as rnd
from rlbox.rand import cache_gen
from .tiles import IHT, tiles


class SemiGradientSarsa(AgentProgram):
    MAX_SIZE = 4096
    N_TILINGS = 8

    def __init__(self, act_spec: Spec, obs_spec: Spec, alpha=0.5, epsilon=0.1,
                 gamma=1.0):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.hashtable = IHT(self.MAX_SIZE)
        self.w = np.zeros(self.MAX_SIZE)
        self.scales = [self.N_TILINGS / (s.hi - s.lo) for s in obs_spec]
        self.step_size = alpha / self.N_TILINGS

        self.sample = cache_gen(
            lambda: np.random.randint(0, act_spec.size(), 1000))

        self.s = None
        self.a = None

    def reset(self):
        self.s = None
        self.a = None

    def _tiles(self, obs, action):
        return tiles(self.hashtable, self.N_TILINGS,
                     [obs[i] * self.scales[i] for i, s in
                      enumerate(self.obs_spec)],
                     [action])

    def __call__(self, obs, reward, done):
        if self.epsilon > rnd.rand():
            action = next(self.sample)
        else:
            _q = np.array([self.q(obs, a, done)
                           for a in range(self.act_spec.size())])
            action = np.random.choice(np.flatnonzero(_q == _q.max()))

        if self.s is not None and self.a is not None:
            self.learn(self.s, self.a, reward, obs, action, done)

        self.s = obs
        self.a = action

        return action

    def q(self, s, a, done):
        if done:
            return 0.0
        return np.sum(self.w[self._tiles(s, a)])

    def learn(self, s0, a0, r, s1, a1, done):
        q0 = self.q(s0, a0, False)
        q1 = self.q(s1, a1, done)
        delta = self.step_size * (r + self.gamma * q1 - q0)
        self.w[self._tiles(s0, a0)] += delta

    def cost_to_go(self, obs):
        return -np.max(np.array([self.q(obs, a, False)
                                 for a in range(self.act_spec.size())]))


class TrueOnlineSarsaLambda(AgentProgram):
    MAX_SIZE = 4096
    N_TILINGS = 8

    def __init__(self, act_spec: Spec, obs_spec: Spec, alpha=0.5, epsilon=0.1,
                 gamma=1.0, lmbda=0.9):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.lmbda = lmbda
        self.hashtable = IHT(self.MAX_SIZE)
        self.w = np.zeros(self.MAX_SIZE)
        self.z = np.zeros(self.MAX_SIZE)
        self.scales = [self.N_TILINGS / (s.hi - s.lo) for s in obs_spec]
        self.step_size = alpha / self.N_TILINGS
        self.q_old = 0

        self.sample = cache_gen(
            lambda: np.random.randint(0, act_spec.size(), 1000))

        self.s = None
        self.a = None

    def reset(self):
        self.s = None
        self.a = None
        self.z = np.zeros(self.MAX_SIZE)
        self.q_old = 0

    def _tiles(self, obs, action):
        return tiles(self.hashtable, self.N_TILINGS,
                     [obs[i] * self.scales[i] for i, s in
                      enumerate(self.obs_spec)],
                     [action])

    def __call__(self, obs, reward, done):
        if self.epsilon > rnd.rand():
            action = next(self.sample)
        else:
            _q = np.array([self.q(obs, a, done)
                           for a in range(self.act_spec.size())])
            action = np.random.choice(np.flatnonzero(_q == _q.max()))

        if self.s is not None and self.a is not None:
            self.learn(self.s, self.a, reward, obs, action, done)

        self.s = obs
        self.a = action

        return action

    def q(self, s, a, done):
        if done:
            return 0.0
        return np.sum(self.w[self._tiles(s, a)])

    def learn(self, s0, a0, r, s1, a1, done):
        q0 = self.q(s0, a0, False)
        q1 = self.q(s1, a1, done)
        active_tiles = self._tiles(s0, a0)
        self.dutch_trace(active_tiles)
        delta = r + self.gamma * q1 - q0
        self.w += self.step_size * (delta + q0 - self.q_old) * self.z
        self.w[active_tiles] -= self.step_size * (q0 - self.q_old)
        self.q_old = q1

    def dutch_trace(self, active_tiles):
        coef = (1 - self.step_size * self.gamma * self.lmbda *
                np.sum(self.z[active_tiles]))
        self.z *= self.gamma * self.lmbda
        self.z[active_tiles] += coef

    def cost_to_go(self, obs):
        return -np.max(np.array([self.q(obs, a, False)
                                 for a in range(self.act_spec.size())]))


class ActorCriticLambda(AgentProgram):
    MAX_SIZE = 4096
    N_TILINGS = 8

    def __init__(self, act_spec: Spec, obs_spec: Spec,
                 alpha_w=0.5, alpha_theta=0.5,
                 gamma=1.0, lambda_w=0.9, lambda_theta=0.9):
        AgentProgram.__init__(self, act_spec, obs_spec)

        # critic
        self.w = np.zeros(self.MAX_SIZE)
        self.alpha_w = alpha_w
        self.lambda_w = lambda_w
        self.z_w = np.zeros(self.MAX_SIZE)
        self.step_size_w = alpha_w / self.N_TILINGS

        # actor
        self.theta = np.zeros(self.MAX_SIZE)
        self.alpha_theta = alpha_theta
        self.lambda_theta = lambda_theta
        self.z_theta = np.zeros(self.MAX_SIZE)
        self.step_size_theta = alpha_theta / self.N_TILINGS

        self.gamma = gamma
        self.hashtable = IHT(self.MAX_SIZE)
        self.scales = [self.N_TILINGS / (s.hi - s.lo) for s in obs_spec]
        self._a = np.arange(act_spec.size()).tolist()

        self.s0 = None
        self.a0 = None
        self.I = 1

    def reset(self):
        self.s0 = None
        self.a0 = None
        self.z_w = np.zeros(self.MAX_SIZE)
        self.z_theta = np.zeros(self.MAX_SIZE)
        self.I = 1

    def __call__(self, obs, reward, done):
        if self.a0 is not None:
            self.learn(self.s0, self.a0, reward, obs, done)

        self.s0 = obs
        self.a0 = np.random.choice(self._a, p=self.policy(obs))

        return self.a0

    def policy(self, s):
        return self._softmax(
            [np.sum(self.theta[self._tiles(s, [a])]) for a in self._a])

    def _tiles(self, obs, action=[]):
        return tiles(self.hashtable, self.N_TILINGS,
                     [obs[i] * self.scales[i] for i, s in
                      enumerate(self.obs_spec)],
                     action)

    @staticmethod
    def _softmax(arr):
        return np.exp(arr) / np.sum(np.exp(arr), axis=0)

    def learn(self, s0, a0, r, s1, done):
        v0 = self.v(s0, False)
        v1 = self.v(s1, done)
        delta = r + self.gamma * v1 - v0
        self.dutch_trace_w(s0)
        self.acc_trace_theta(s0, a0)
        self.w += self.step_size_w * delta * self.z_w
        self.theta += self.step_size_theta * delta * self.z_theta
        self.I *= self.gamma

    def v(self, s, done):
        if done:
            return 0.0
        return np.sum(self.w[self._tiles(s)])

    def x(self, s, a):
        x = np.zeros(self.MAX_SIZE)
        x[self._tiles(s, [a])] = 1
        return x

    def dutch_trace_w(self, s):
        active_tiles = self._tiles(s)
        coef = (1 - self.step_size_w * self.gamma * self.lambda_w *
                np.sum(self.z_w[active_tiles]))

        self.z_w *= self.gamma * self.lambda_w
        self.z_w[active_tiles] += coef

    def acc_trace_theta(self, s, a):
        grad = self.x(s, a)
        for b, p in enumerate(self.policy(s)):
            grad -= p * self.x(s, b)
        self.z_theta *= self.gamma * self.lambda_theta
        self.z_theta += self.I * grad

    def cost_to_go(self, obs):
        return -self.v(obs, False)
