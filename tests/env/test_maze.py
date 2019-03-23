from rlbox.env.maze import Maze


def test_prepares_action_space():
    env = Maze()
    assert env.act_spec[0].shape == (1,)
    assert env.act_spec[0].lo == 0
    assert env.act_spec[0].hi == 3


def test_prepares_observation_space():
    env = Maze()
    assert env.obs_spec[0].shape == (1,)
    assert env.obs_spec[0].lo == 0
    assert env.obs_spec[0].hi == 8
    assert env.obs_spec[1].shape == (1,)
    assert env.obs_spec[1].lo == 0
    assert env.obs_spec[1].hi == 5


def test_does_not_allow_step_outside_gird():
    env = Maze()
    env.s = [0, 5]
    state, reward = env.step(0)

    assert state == [0, 5]
    assert reward == 0

    env.s = [0, 0]
    state, reward = env.step(1)

    assert state == [0, 0]
    assert reward == 0


def test_is_done_when_goal_is_reached():
    env = Maze()
    env.s = [8, 4]
    env.step(0)

    assert env.done()
    assert env.reward == 1


def test_does_not_allow_to_step_on_wall():
    env = Maze()
    env.s = [2, 1]
    state, reward = env.step(0)

    assert state == [2, 1]
    assert reward == 0
