import itertools
from unittest.mock import patch

import numpy as np

from rlbox.env.car_rental import CarRentalEnv


def env():
    return CarRentalEnv(max_move=5, max_cars=20, expct=[3, 4, 3, 2],
                        s0=[10, 10])


class TestCarRentalEnv:
    def test_prepares_action_space(self):
        assert env().act_spc == {'low': -5, 'high': 5, 'shape': (1,)}

    def test_prepares_observation_space(self):
        assert env().obs_spc == {'low': 0, 'high': 20, 'shape': (2,)}

    def test_on_step_moves_car_between_locations(self):
        with patch('rlbox.env.car_rental.np.random') as rnd:
            rnd.poisson.return_value = 0

            obs, reward = env().step(3)

            assert obs == [7, 13]
            assert reward == -6

    def test_on_step_trims_too_big_move_action(self):
        with patch('rlbox.env.car_rental.np.random') as rnd:
            rnd.poisson.return_value = 0

            obs, reward = env().step(-13)

            assert obs == [20, 0]
            assert reward == -26

    def test_on_step_rents_cars(self):
        with patch('rlbox.env.car_rental.np.random') as rnd:
            rnd.poisson.side_effect = [2, 11, 0, 0]

            obs, reward = env().step(0)

            assert obs == [8, 0]
            assert reward == 120

    def test_on_step_returns_cars(self):
        with patch('rlbox.env.car_rental.np.random') as rnd:
            rnd.poisson.side_effect = [0, 0, 11, 4]

            obs, reward = env().step(0)

            assert obs == [20, 14]
            assert reward == 0

    def test_provides_mdp_model(self):
        car_env = CarRentalEnv(max_move=1, max_cars=1,
                               expct=[0.001, 0.001, 0.001, 0.001],
                               s0=[0, 0])
        car_env.prepare_model()
        assert car_env.P.shape == (4, 4, 3)
        assert car_env.R.shape == (4, 4, 3)
        for state, action in itertools.product(range(0, 4), range(0, 3)):
            assert np.isclose(np.sum(car_env.P[state, :, action]), 1.0)
