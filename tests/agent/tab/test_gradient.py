import numpy as np

import rlbox.core as core
from rlbox.agent.tab.gradient import GradientBandit


class TestGradientBandit:
    act_spec = core.Spec([core.Space(shape=(10,))])
    obs_spec = core.Spec([core.Space(shape=(10,))])

    def test_distributes_policy_with_softmax(self):
        agent = GradientBandit(self.act_spec, self.obs_spec,
                               alpha=0.2)

        assert np.all(agent.policy() == np.full(10, 0.1))

    def test_updates_baseline(self):
        agent = GradientBandit(self.act_spec, self.obs_spec,
                               alpha=0.2, baseline=True)

        agent(1, 2.0)
        assert agent.rbase() != 0

    def test_does_not_update_baseline_if_disabled(self):
        agent = GradientBandit(self.act_spec, self.obs_spec,
                               alpha=0.2, baseline=False)

        agent(1, 2.0)
        assert not agent.rbase()

    def test_updates_policy(self):
        agent = GradientBandit(self.act_spec, self.obs_spec,
                               alpha=0.2)

        agent(1, 2.0)
        old_policy = agent.policy()
        agent(4, 8.0)

        assert np.sum(old_policy) == 1
        assert np.sum(agent.policy()) == 1
        assert np.any(old_policy != agent.policy())
