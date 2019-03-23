import rlbox.core as core
import rlbox.agent.tab.epsilongreedy as epsilongreedy
from rlbox.env.narmedbandit import NArmedBanditEnv


class TestRun:
    def test_collects_reward_from_each_step(self):
        arms = 10
        eps = 0.1
        environment = NArmedBanditEnv(10, arms)
        agent = core.Agent(
            epsilongreedy.SampleAverage(
                act_spec=core.Spec([core.Space(shape=(10,))]),
                obs_spec=core.Spec([core.Space(shape=(10,))]),
                epsilon=eps),
            lambda env: (env.last_action, env.reward),
            lambda action, env: env.step(action))

        core.Run(agent, environment).start()
        assert len(environment.all_rewards) == 10
        assert len(environment.optimal_actions) == 10
