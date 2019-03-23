from enum import Enum

import rlbox.agent.approx.linear as linear
import rlbox.agent.tab.dyna as dyna
import rlbox.agent.tab.epsilongreedy as eps_greed
import rlbox.agent.tab.gradient as gradient
import rlbox.agent.tab.random as random
import rlbox.agent.tab.ucb as ucb


class Algorithm(Enum):
    SMPL_AVG = 'smpl_avg'
    WEIGHT_AVG = 'weight_avg'
    WEIGHT_AVG_NBIAS = 'weight_avg_nbias'
    UCB = 'ucb1'
    GRADIENT = 'grad_bandit'
    RANDOM = 'random'
    DYNA_Q = 'dynaq'
    DYNA_Q_V2 = 'dynaqv2'
    SEMIGRADIENT_SARSA = 'semigradient_sarsa'
    TRUE_SARSA_LAMBDA = 'true_sarsa_lambda'
    ACTOR_CRITIC = 'actor_critic'

    def __str__(self):
        return str(self.value)


AGENT_PROGRAM = {
    Algorithm.SMPL_AVG: eps_greed.SampleAverage,
    Algorithm.WEIGHT_AVG: eps_greed.WeightedAverage,
    Algorithm.WEIGHT_AVG_NBIAS: eps_greed.WeightedAverageNBias,
    Algorithm.UCB: ucb.Ucb1,
    Algorithm.GRADIENT: gradient.GradientBandit,
    Algorithm.RANDOM: random.RandomAgent,
    Algorithm.DYNA_Q: dyna.DynaQ,
    Algorithm.DYNA_Q_V2: dyna.DynaQv2,
    Algorithm.SEMIGRADIENT_SARSA: linear.SemiGradientSarsa,
    Algorithm.TRUE_SARSA_LAMBDA: linear.TrueOnlineSarsaLambda,
    Algorithm.ACTOR_CRITIC: linear.ActorCriticLambda
}
