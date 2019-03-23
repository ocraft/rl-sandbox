from rlbox.env.windy_gridworld import WindyGridWorld


def test_prepares_action_space():
    env = WindyGridWorld()
    assert env.act_spec[0].shape == (1,)
    assert env.act_spec[0].lo == 0
    assert env.act_spec[0].hi == 7


def test_prepares_observation_space():
    env = WindyGridWorld()
    assert env.obs_spec[0].shape == (1,)
    assert env.obs_spec[0].lo == 0
    assert env.obs_spec[0].hi == 9
    assert env.obs_spec[1].shape == (1,)
    assert env.obs_spec[1].lo == 0
    assert env.obs_spec[1].hi == 6


def test_moves_with_the_wind():
    env = WindyGridWorld()
    env.s = [3, 3]
    state, reward = env.step(2)

    assert state == [4, 4]
    assert reward == -1
    assert not env.done()


def test_does_not_allow_step_outside_gird():
    env = WindyGridWorld()
    env.s = [9, 6]
    state, reward = env.step(5)

    assert state == [9, 6]
    assert reward == -1

    env.s = [0, 0]
    state, reward = env.step(7)

    assert state == [0, 0]
    assert reward == -1


def test_is_done_when_goal_is_reached():
    env = WindyGridWorld()
    env.s = [8, 2]
    env.step(3)

    assert env.done()
