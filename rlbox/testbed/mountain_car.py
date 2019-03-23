import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

from rlbox.agent.approx.linear import SemiGradientSarsa
import rlbox.core as core
from rlbox.env.mountain_car import MountainCar
from rlbox.testbed.config import AGENT_PROGRAM
from .plot import labels_for
from .sink import SharedMem, Store
from .testbed import Testbed


class MountainCarTestbed(Testbed):
    def __init__(self, runs, env, exe):
        np.random.seed(10)
        Testbed.__init__(self, runs, env, exe, r"mountain_car")

    def sink_cfg(self):
        cfg = {}
        run_cfg = {'runs': self.runs}
        for task in self.exe:
            params = {**self.env, **run_cfg, **{'alg': task[0]}, **task[1],
                      **task[2]}
            key_ep = self.key_for('episodes', **params)
            if key_ep not in cfg:
                cfg[key_ep] = (self.runs, task[2]['episodes'], 'i')
        return cfg

    @staticmethod
    def worker(runs, env_cfg, task, param):
        environment = MountainCar(**env_cfg)
        agent_program = AGENT_PROGRAM[task[0]](environment.act_spec,
                                               environment.obs_spec,
                                               **task[1])
        agent = core.Agent(
            agent_program,
            lambda env: (env.obs, env.reward, env.done()),
            lambda action, env: env.step(action))

        steps = []
        for _ in range(0, task[2]['episodes']):
            core.Run(agent, environment).start()

            steps.append(environment.nstep)

            agent_program.reset()
            environment.reset()

        key = {**env_cfg, **{'runs': runs}, **{'alg': task[0]}, **task[1],
               **task[2]}
        SharedMem.dump(Testbed.key_for('episodes', **key), param['run'], steps)

    def plot(self):
        plt.style.use('seaborn')

        run_cfg = {'runs': self.runs}
        store = Store(self.summary)
        labels = [labels_for(**{'alg': e[0]}, **e[1]) for e in self.exe]
        for task in self.exe:
            params = {**self.env, **run_cfg, **{'alg': task[0]}, **task[1],
                      **task[2]}

            (pd.DataFrame(store[self.key_for('episodes', **params)][:])
             .mean()
             .plot())

        plt.xlabel('Episodes')
        plt.ylabel('Steps per episode')
        plt.legend(loc='best', labels=labels)

        store.close()

        environment = MountainCar()
        agent_program = AGENT_PROGRAM[self.exe[0][0]](environment.act_spec,
                                                      environment.obs_spec,
                                                      **self.exe[0][1])
        agent = core.Agent(
            agent_program,
            lambda env: (env.obs, env.reward, env.done()),
            lambda action, env: env.step(action))
        core.Run(agent, environment).start()

        fig = plt.figure()
        grid_size = 40
        ax = fig.gca(projection='3d')
        pos = np.linspace(environment.BOUND_MIN_P, environment.BOUND_MAX_P,
                          grid_size)
        vel = np.linspace(environment.BOUND_MIN_V, environment.BOUND_MAX_V,
                          grid_size)

        axis_x = []
        axis_y = []
        axis_z = []
        for p in pos:
            for v in vel:
                axis_x.append(p)
                axis_y.append(v)
                axis_z.append(agent_program.cost_to_go((p, v)))

        ax.scatter(axis_x, axis_y, axis_z)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Cost to go')
        ax.set_title('After first episode')
        plt.show()
