import numpy as np

from rlbox.core import AgentProgram
from rlbox.core.space import Spec
from rlbox.rand import cache_gen


class RandomAgent(AgentProgram):

    def __init__(self, act_spec: Spec, obs_spec: Spec, pi=None):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self.sample = cache_gen(lambda: self.act_spec[0].sample(1000))
        self.pi = pi

    def __call__(self, obs, reward):
        if self.pi is not None:
            return self.pi[obs]
        return next(self.sample)

    def policy(self):
        act_size = self.act_spec.size()
        return np.full((self.obs_spec.size(), act_size),
                       fill_value=1 / act_size)
