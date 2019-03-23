from rlbox.env.mountain_car import MountainCar


def test_prepares_action_space():
    env = MountainCar()
    assert env.act_spec[0].shape == (1,)
    assert env.act_spec[0].lo == -1
    assert env.act_spec[0].hi == 1


def test_prepares_observation_space():
    env = MountainCar()
    assert env.obs_spec[0].shape == (1,)
    assert env.obs_spec[0].lo == -1.2
    assert env.obs_spec[0].hi == 0.5
    assert env.obs_spec[1].shape == (1,)
    assert env.obs_spec[1].lo == -0.07
    assert env.obs_spec[1].hi == 0.07
    assert env.obs_spec[0].is_continuous()
    assert env.obs_spec[1].is_continuous()


def test_is_done_when_top_of_the_mountain_is_reached():
    env = MountainCar()
    env.p = 0.5

    assert env.done()


def test_bounds_position():
    env = MountainCar()
    env.p = -1.19
    env.v = env.BOUND_MIN_V
    env.step(0)

    assert env.p == env.BOUND_MIN_P
    assert env.v == 0.0
    assert env.reward == -1


def test_bounds_velocity():
    env = MountainCar()
    env.v = env.BOUND_MIN_V
    env.step(0)

    assert env.v == env.BOUND_MIN_V

    env = MountainCar()
    env.v = env.BOUND_MAX_V
    env.step(2)

    assert env.v == env.BOUND_MAX_V
