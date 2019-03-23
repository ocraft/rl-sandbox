import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tqdm

import rlbox.core as core
from rlbox.agent.tab.random import RandomAgent
from rlbox.algorithm import off_policy_monte_carlo
from rlbox.env.racetrack import RaceTrack
from rlbox.testbed.config import AGENT_PROGRAM
from .sink import SharedMem, Store
from .testbed import Testbed


class RaceTrackTestbed(Testbed):
    def __init__(self, runs, env, exe):
        Testbed.__init__(self, runs, env, exe, r"racetrack")

    def sink_cfg(self):
        cfg = {}
        run_cfg = {'runs': self.runs}
        for task in self.exe:
            params = {**self.env, **run_cfg, **{'alg': task[0]}, **task[1]}
            key_st = self.key_for('states', **params)
            key_rew = self.key_for('rewards', **params)
            key_act = self.key_for('actions', **params)
            if key_st not in cfg:
                cfg[key_st] = (self.runs, self.env['steps'], 'i')
            if key_rew not in cfg:
                cfg[key_rew] = (self.runs, self.env['steps'], 'i')
            if key_act not in cfg:
                cfg[key_act] = (self.runs, self.env['steps'], 'i')
        return cfg

    @staticmethod
    def worker(runs, env_cfg, task, param):
        environment = RaceTrack(**env_cfg)
        agent = core.Agent(
            AGENT_PROGRAM[task[0]](
                environment.act_spec,
                environment.obs_spec,
                **task[1]),
            lambda env: (env.obs, -1),
            lambda action, env: env.step(action))

        core.Run(agent, environment).start()

        key = {**env_cfg, **{'runs': runs}, **{'alg': task[0]}, **task[1]}
        states, actions, rewards = environment.episode()
        run = param['run']
        SharedMem.dump(Testbed.key_for('states', **key), run, states)
        SharedMem.dump(Testbed.key_for('rewards', **key), run, rewards)
        SharedMem.dump(Testbed.key_for('actions', **key), run, actions)

    def plot(self):
        store = Store(self.summary)

        params = {**self.env, **{'runs': self.runs},
                  **{'alg': self.exe[0][0]}, **self.exe[0][1]}
        st = store[self.key_for('states', **params)]
        act = store[self.key_for('actions', **params)]
        rew = store[self.key_for('rewards', **params)]

        environment = RaceTrack(**self.env)

        if self.key_for('policy', **params) in store:
            pi = store[self.key_for('policy', **params)][:]
        else:
            agent_prog = RandomAgent(environment.act_spec, environment.obs_spec)
            pi = off_policy_monte_carlo(
                environment.act_spec,
                environment.obs_spec,
                agent_prog.policy(),
                stream(st, act, rew))
        store.replace(self.key_for('policy', **params), pi)
        store.close()
        agent = core.Agent(
            RandomAgent(environment.act_spec, environment.obs_spec, pi),
            lambda env: (environment.STATE_IDX[env.obs], -1),
            lambda action, env: env.step(environment.ACT_SPACE[action]))

        def step(_):
            if not environment.done():
                agent.run(environment)
                im.set_array(environment.print())
            return im,

        for _ in range(0, 5):
            fig = plt.figure()
            environment = RaceTrack(**self.env)
            im = plt.imshow(environment.print(),
                            origin='lower', interpolation='none',
                            animated=True)
            plt.gca().invert_yaxis()

            ani = animation.FuncAnimation(fig, step, interval=100, blit=True)
            plt.show()


def stream(st, act, rew):
    i = 0
    batch = 1000
    progress_bar = tqdm.tqdm(len(st))
    while i < len(st):
        episodes = [list(zip(i, j, k)) for i, j, k in
                    zip(st[i:i + batch], act[i:i + batch], rew[i:i + batch])]
        for episode in episodes:
            yield episode
        i += batch
        progress_bar.update(batch)
    progress_bar.close()
