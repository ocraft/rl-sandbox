from unittest.mock import patch

import numpy as np

from rlbox.agent.tab.td import OnPolicySarsa
from rlbox.agent.tab.td import e_greedy_policy, OnPolicyNStepSarsa
import rlbox.core as core
from rlbox.env.windy_gridworld import WindyGridWorld


class TestOnPolicySarsa:
    def test_updates_q_with_whole_quintuple_of_events(self):
        with patch('rlbox.env.racetrack.rnd') as rnd:
            rnd.rand.return_value = 1
            agent = OnPolicySarsa(
                act_spec=core.Spec([core.Space(shape=(1, 0))]),
                obs_spec=core.Spec([core.Space(shape=(1, 0))])
            )
            agent(0, -1)
            agent(1, -1)

            assert np.all(agent.q == [[-0.5, 0], [0, 0]])


def test_egreedy_policy():
    assert np.all(np.isclose(
        e_greedy_policy(np.zeros(shape=(2, 3)), np.zeros(shape=(2, 3)), 0.1),
        np.array([[0.9333333, 0.0333333, 0.0333333],
                  [0.9333333, 0.0333333, 0.0333333]])))


def test_nstep_sarsa():
    environment = WindyGridWorld()
    agent_program = OnPolicyNStepSarsa(
        environment.act_spec,
        environment.obs_spec,
        n=2,
        alpha=0.5,
        epsilon=0.1,
        gamma=1.0)
    agent = core.Agent(
        agent_program,
        lambda env: (env.STATE_IDX[env.obs], env.reward, env.done()),
        lambda action, env: env.step(action))
    core.Run(agent, environment).start()
