import fractions

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

import rlbox.core as core
from rlbox.env.narmedbandit import NArmedBanditEnv
from rlbox.testbed.config import AGENT_PROGRAM
from .config import Algorithm
from .plot import bandit_problem_plt, average_reward_plt, optimal_action_plt
from .plot import labels_for, prettify
from .sink import SharedMem, Store
from .testbed import Testbed


class NArmedBanditTestbed(Testbed):
    def __init__(self, runs, env, exe):
        Testbed.__init__(self, runs, env, exe, r"narmedbandit")

    def sink_cfg(self):
        cfg = {}
        run_cfg = {'runs': self.runs}
        for task in self.exe:
            params = {**self.env, **run_cfg, **{'alg': task[0]}, **task[1]}
            key_rew = self.key_for('rewards', **params)
            key_act = self.key_for('actions', **params)
            if key_rew not in cfg:
                cfg[key_rew] = (self.runs, self.env['steps'], 'd')
            if key_act not in cfg:
                cfg[key_act] = (self.runs, self.env['steps'], 'i')
        return cfg

    @staticmethod
    def worker(runs, env_cfg, task, param):
        environment = NArmedBanditEnv(**env_cfg)
        agent = core.Agent(
            AGENT_PROGRAM[task[0]](
                environment.act_spec,
                environment.obs_spec,
                **task[1]),
            lambda env: (env.last_action, env.reward),
            lambda action, env: env.step(action))

        core.Run(agent, environment).start()

        key = {**env_cfg, **{'runs': runs}, **{'alg': task[0]}, **task[1]}
        SharedMem.dump(
            Testbed.key_for('rewards', **key),
            param['run'],
            environment.all_rewards)
        SharedMem.dump(
            Testbed.key_for('actions', **key),
            param['run'],
            environment.optimal_actions)

    def plot(self):
        plt.style.use('seaborn')
        fig, axes = plt.subplots(3, figsize=(8, 10))
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(
            r'''
            N-Armed Bandit: {}
            '''.format(labels_for(**self.env)), fontsize=12)

        run_cfg = {'runs': self.runs}
        store = Store(self.summary)
        for task in self.exe:
            params = {**self.env, **run_cfg, **{'alg': task[0]}, **task[1]}

            (pd.DataFrame(store[self.key_for('rewards', **params)][:])
             .mean()
             .plot(ax=axes[1]))

            (pd.DataFrame(store[self.key_for('actions', **params)][:])
             .mean()
             .plot(ax=axes[2]))

        bandit_problem_plt(axis=axes[0], **self.env)

        labels = [labels_for(**{'alg': e[0]}, **e[1]) for e in self.exe]
        average_reward_plt(axis=axes[1], label=labels)
        optimal_action_plt(axis=axes[2], label=labels)

        store.close()
        plt.show()


class NArmedBanditParamStudy(NArmedBanditTestbed):
    def plot(self):
        plt.style.use('seaborn')
        fig, axis = plt.subplots(1, figsize=(8, 5))
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(
            r'''
            N-Armed Bandit: {}
            '''.format(labels_for(**self.env)), fontsize=12)

        run_cfg = {'runs': self.runs}

        param_to_study = {
            Algorithm.SMPL_AVG: 'epsilon',
            Algorithm.WEIGHT_AVG: 'bias',
            Algorithm.GRADIENT: 'alpha',
            Algorithm.UCB: 'c'
        }

        param_study = {}
        xticks = set()
        labels = set()
        store = Store(self.summary)
        for task in self.exe:
            params = {**self.env, **run_cfg, **{'alg': task[0]}, **task[1]}

            perf = (pd.DataFrame(store[self.key_for('rewards', **params)][:])
                    .mean(axis=1)
                    .mean())

            param_name = param_to_study[task[0]]
            param = task[1][param_name]

            if task[0] not in param_study:
                param_study[task[0]] = {}
            param_study[task[0]][param] = perf
            xticks.add(param)
            if task[0] not in labels:
                labels.add(str(task[0]) + '({0})'.format(prettify(param_name)))

        pd.DataFrame(param_study).plot(ax=axis, logx=True)
        axis.set_xticks(sorted(xticks))
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: fractions.Fraction(x)))
        axis.set_xlabel('Param value')
        axis.set_ylabel('Average reward over first 1000 steps')

        axis.legend(loc='best', labels=labels)
        axis.autoscale()

        store.close()
        plt.show()
