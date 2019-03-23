import numpy as np

from rlbox.algorithm.policy_iteration import value, policy_evaluation
from rlbox.algorithm.policy_iteration import policy_improvement
from rlbox.algorithm.policy_iteration import policy_iteration, value_iteration
from rlbox.env.mdp import Mdp


class SampleEnv(Mdp):
    def __init__(self):
        Mdp.__init__(self, 0)
        self._S = np.arange(2)
        self._A = np.arange(2)
        self._R = np.array([
            [[5, 0], [5, 10]],
            [[0, 0], [-1, 0]]])
        self._P = np.array([
            [[0.5, 0.0], [0.5, 1.0]],
            [[0.0, 0.0], [1.0, 0.0]]])

    def prepare_model(self):
        pass


def test_computes_value_of_state():
    env = SampleEnv()
    V = np.zeros(shape=env.S.shape)

    assert np.isclose(value(env, np.array([1, 0]), 0, V, 1.0), 5.0)


def test_evaluates_policy():
    env = SampleEnv()
    V = np.zeros(shape=env.S.shape)

    policy = [1, 0]

    policy_evaluation(env, policy, V, 0.9)

    assert np.all(np.isclose(V, [1.0872796, -9.91272036]))


def test_improves_policy():
    env = SampleEnv()
    V = np.zeros(shape=env.S.shape)

    policy = [0, 0]

    policy_improvement(env, policy, V, 0.9)

    assert policy == [1, 0]


def test_finds_optimal_policy():
    env = SampleEnv()

    policy, V = policy_iteration(
        env, np.zeros(env.S.shape, dtype=np.int32), 0.9)

    assert np.all(policy == [[0, 0], [1, 0]])
    assert np.all(np.isclose(V, [1.0706965, -9.9293035]))


def test_find_optimal_values():
    env = SampleEnv()

    policy, values = value_iteration(env, 0.9)

    assert np.all(policy == [1, 0])
    assert np.all(np.isclose(values[-1], [1.09697737, -9.90302263]))
