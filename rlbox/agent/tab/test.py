import numpy as np

from rlbox.core import AgentProgram
from rlbox.core.space import Spec


class TestAgent(AgentProgram):

    def __init__(self, act_spec: Spec, obs_spec: Spec, pi):
        AgentProgram.__init__(self, act_spec, obs_spec)
        self.pi = pi
        self.actions = np.arange(act_spec.size())

    def __call__(self, obs, reward):
        return np.random.choice(a=self.actions, p=self.pi[obs, :])
