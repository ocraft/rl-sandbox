import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import rlbox.core as core
from rlbox.env.maze import Maze
from rlbox.testbed.config import AGENT_PROGRAM
from .plot import labels_for
from .sink import SharedMem, Store
from .testbed import Testbed


class MazeTestbed(Testbed):
    def __init__(self, runs, env, exe):
        np.random.seed(10)
        Testbed.__init__(self, runs, env, exe, r"maze")

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
        environment = Maze(**env_cfg)
        agent_program = AGENT_PROGRAM[task[0]](environment.act_spec,
                                               environment.obs_spec,
                                               **task[1])
        agent = core.Agent(
            agent_program,
            lambda env: (env.STATE_IDX[env.obs], env.reward),
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
        plt.show()


class BlockingMazeTestbed(Testbed):
    def __init__(self, runs, env, exe):
        np.random.seed(10)
        Testbed.__init__(self, runs, env, exe, r"maze")

    def sink_cfg(self):
        cfg = {}
        run_cfg = {'runs': self.runs}
        for task in self.exe:
            params = {**self.env, **run_cfg, **{'alg': task[0]}, **task[1],
                      **task[2]}
            key_rew = self.key_for('rewards', **params)
            if key_rew not in cfg:
                cfg[key_rew] = (self.runs, task[2]['steps'], 'i')
        return cfg

    @staticmethod
    def worker(runs, env_cfg, task, param):
        environment = Maze(**env_cfg)
        agent_program = AGENT_PROGRAM[task[0]](environment.act_spec,
                                               environment.obs_spec,
                                               **task[1])
        agent = core.Agent(
            agent_program,
            lambda env: (env.STATE_IDX[env.obs], env.reward),
            lambda action, env: env.step(action))

        while environment.steps_cnt < task[2]['steps']:
            core.Run(agent, environment).start()

            agent_program.reset()
            environment.reset()

        key = {**env_cfg, **{'runs': runs}, **{'alg': task[0]}, **task[1],
               **task[2]}
        SharedMem.dump(Testbed.key_for('rewards', **key), param['run'],
                       environment.rewards[0: task[2]['steps']])

    def plot(self):
        plt.style.use('seaborn')

        run_cfg = {'runs': self.runs}
        store = Store(self.summary)
        labels = [labels_for(**{'alg': e[0]}, **e[1]) for e in self.exe]
        for task in self.exe:
            params = {**self.env, **run_cfg, **{'alg': task[0]}, **task[1],
                      **task[2]}

            (pd.DataFrame(store[self.key_for('rewards', **params)][:])
             .mean()
             .cumsum()
             .plot())

        plt.xlabel('Time steps')
        plt.ylabel('Cumulative rewards')
        plt.legend(loc='best', labels=labels)

        store.close()
        plt.show()
