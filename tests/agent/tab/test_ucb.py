import rlbox.core as core
from rlbox.agent.tab.ucb import Ucb1


class TestUcb1:
    def test_updates_action_value_with_sample_average_reward(self):
        agent = Ucb1(
            core.Spec([core.Space(shape=(10,))]),
            core.Spec([core.Space(shape=(10,))]),
            c=2)

        action = agent(None, 0)
        action = agent(action, -2.0)
        action = agent(action, 3.0)
        agent(action, 1.0)

        assert agent.qtable()[0] == -2.0
        assert agent.qtable()[1] == 3.0
        assert agent.qtable()[2] == 1.0

    def test_takes_best_action_using_action_value_and_confidence(self):
        agent = Ucb1(
            core.Spec([core.Space(shape=(2,))]),
            core.Spec([core.Space(shape=(2,))]),
            c=2)

        action = agent(None, 0)
        assert action == 0
        action = agent(action, 2.0)
        assert action == 1
        action = agent(action, 3.0)
        assert action == 1
        action = agent(action, 0.5)
        assert action == 0
