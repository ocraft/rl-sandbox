from abc import abstractmethod, ABC

from rlbox.core.space import Spec


class Environment(ABC):
    def __init__(self, act_spec: Spec, obs_spec: Spec):
        self.act_spec = act_spec
        self.obs_spec = obs_spec

    @abstractmethod
    def done(self):
        pass


class AgentProgram(ABC):
    def __init__(self, act_spec: Spec, obs_spec: Spec):
        self.act_spec = act_spec
        self.obs_spec = obs_spec


class Agent:
    def __init__(self, program, sensor, actuator):
        self.sensor = sensor
        self.actuator = actuator
        self.agent_program = program

    def run(self, env):
        self.actuator(self.agent_program(*self.sensor(env)), env)

    def terminal(self, env):
        self.agent_program(*self.sensor(env))


class Run:
    def __init__(self, agent, env):
        self.env = env
        self.agent = agent

    def start(self):
        while not self.env.done():
            self.agent.run(self.env)
        self.agent.terminal(self.env)
