from unittest.mock import patch

import rlbox.core as core
from rlbox.agent.tab.epsilongreedy import SampleAverage, WeightedAverage


class TestSampleAverage:
    act_spec = core.Spec([core.Space(shape=(10,))])
    obs_spec = core.Spec([core.Space(shape=(10,))])

    def test_explores_with_epsilon_probability(self):
        with patch('rlbox.agent.tab.epsilongreedy.rnd') as sampler:
            sampler.rand.return_value = 0.05
            sampler.rint.return_value = 3

            agent = SampleAverage(self.act_spec, self.obs_spec, 0.1)
            assert agent(None, 0) == 3

    def test_takes_best_action_with_one_less_epsilon_probability(self):
        with patch('rlbox.agent.tab.epsilongreedy.rnd') as sampler:
            sampler.rand.side_effect = [0.05, 0.2]
            sampler.rint.return_value = 3

            agent = SampleAverage(self.act_spec, self.obs_spec,
                                  epsilon=0.1)
            action = agent(None, 0)

            assert agent(action, 2.0) == 3

    def test_updates_action_value_with_sample_average_reward(self):

        with patch('rlbox.agent.tab.epsilongreedy.rnd') as sampler:
            sampler.rand.return_value = 0.2
            agent = SampleAverage(self.act_spec, self.obs_spec,
                                  epsilon=0.1)

            action = agent(None, 0)
            action = agent(action, -2.0)
            action = agent(action, 3.0)
            agent(action, 1.0)

            assert agent.qtable()[0] == -2.0
            assert agent.qtable()[1] == 2.0


class TestWeightedAverage:
    act_spec = core.Spec([core.Space(shape=(10,))])
    obs_spec = core.Spec([core.Space(shape=(10,))])

    def test_updates_action_value_with_weighted_average_reward(self):

        with patch('rlbox.agent.tab.epsilongreedy.rnd') as sampler:
            sampler.rand.return_value = 0.2
            agent = WeightedAverage(self.act_spec, self.obs_spec,
                                    epsilon=0.1,
                                    alpha=0.2)

            action = agent(None, 0)
            action = agent(action, -2.0)
            action = agent(action, 3.0)
            agent(action, 1.0)

            assert agent.qtable()[0] == -0.4
            assert agent.qtable()[1] == 0.68
