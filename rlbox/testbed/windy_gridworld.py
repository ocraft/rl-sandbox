import matplotlib.pyplot as plt
import numpy as np
import tqdm

import rlbox.core as core
from rlbox.agent.tab.td import OnPolicySarsa
from rlbox.env.windy_gridworld import WindyGridWorld


class WindyGridWorldTestbed:
    def __init__(self, runs, stochastic=False, alpha=0.5, epsilon=0.1,
                 gamma=1.0):
        self.runs = runs
        self.steps = []
        self.stochastic = stochastic
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def run(self):
        environment = WindyGridWorld(stochastic=self.stochastic)
        agent_program = OnPolicySarsa(environment.act_spec,
                                      environment.obs_spec,
                                      alpha=self.alpha,
                                      epsilon=self.epsilon,
                                      gamma=self.gamma)
        agent = core.Agent(
            agent_program,
            lambda env: (env.STATE_IDX[env.obs], env.reward),
            lambda action, env: env.step(action))
        for _ in tqdm.tqdm(range(0, self.runs)):
            core.Run(agent, environment).start()

            self.steps.append(environment.nstep)

            agent_program.reset()
            environment.reset()

    def plot(self):
        steps = np.add.accumulate(self.steps)

        plt.plot(steps, np.arange(1, len(steps) + 1))
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.show()
