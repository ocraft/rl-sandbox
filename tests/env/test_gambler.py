import itertools

import numpy as np

from rlbox.env.gambler import Gambler


def test_prepares_mdp_model():
    env = Gambler(0.5, 2)
    env.prepare_model()

    assert env.P.shape == (3, 3, 1)
    assert env.R.shape == (3, 3, 1)
    assert np.isclose(np.sum(env.P[1, :, 0]), 1.0)
    assert env.R[1, 2, 0] == 1
