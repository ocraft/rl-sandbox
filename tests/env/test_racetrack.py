from unittest.mock import patch

import matplotlib.animation as animation
import matplotlib.pyplot as plt

from rlbox.agent.tab.random import RandomAgent
from rlbox.env.racetrack import RaceTrack, TRACK


def test_prepares_action_space():
    track = RaceTrack()
    assert track.act_spec[0].shape == (2,)
    assert track.act_spec[0].lo == -1
    assert track.act_spec[0].hi == 1


def test_prepares_observation_space():
    track = RaceTrack()
    assert len(track.obs_spec) == 4


def test_initializes_on_start_location():
    track = RaceTrack()
    assert TRACK[track.pos[0], track.pos[1]] == RaceTrack.START


def test_initializes_with_zero_velocity():
    track = RaceTrack()
    assert track.vel == [0, 0]


def test_is_done_when_on_end_location():
    track = RaceTrack()
    assert not track.done()
    with patch('rlbox.env.racetrack.rnd') as rnd:
        rnd.rand.return_value = 0
        track.pos = [2, 14]
        track.vel = [1, 3]
        track.step([0, 1])
        assert track.done()


def test_moves_car_with_given_velocity():
    with patch('rlbox.env.racetrack.rnd') as rnd:
        rnd.rand.return_value = 1
        track = RaceTrack()
        start = track.pos.copy()
        state, reward = track.step([1, 0])
        assert state == (start[0] - 1, start[1], 1, 0)
        assert reward == -1


def test_fails_on_accelerate_with_probability():
    with patch('rlbox.env.racetrack.rnd') as rnd:
        rnd.rand.return_value = 0
        track = RaceTrack()
        track.pos = [track.pos[0] - 1, track.pos[1]]
        track.vel = [1, 0]
        pos = track.pos.copy()
        state, reward = track.step([1, 1])
        assert state == (pos[0] - 1, pos[1], 1, 0)
        assert reward == -1


def test_does_not_allow_velocity_out_of_bound():
    with patch('rlbox.env.racetrack.rnd') as rnd:
        rnd.rand.return_value = 1
        track = RaceTrack()
        track.pos = list(track.start_loc[0])
        track.vel = [4, 4]
        state, _ = track.step([1, 1])
        assert state[2:] == (4, 4)


def test_does_not_allow_zero_velocity():
    with patch('rlbox.env.racetrack.rnd') as rnd:
        rnd.rand.return_value = 1
        track = RaceTrack()
        track.pos = list(track.start_loc[0])
        pos = track.pos.copy()
        state, _ = track.step([0, 0])
        assert state == (pos[0] - 1, pos[1] + 1, 1, 1)


def test_returns_to_start_position_if_out():
    with patch('rlbox.env.racetrack.rnd') as rnd:
        rnd.rand.return_value = 1
        track = RaceTrack()
        track.pos = list(track.start_loc[5])
        state, _ = track.step([1, 1])
        assert state[2:] == (0, 0)
        assert track.loc(track.pos) == track.START


def simulate():
    fig = plt.figure()
    track = RaceTrack()
    im = plt.imshow(track.print(),
                    origin='lower', interpolation='none',
                    animated=True)
    plt.gca().invert_yaxis()
    agent = RandomAgent(track.act_spec, track.obs_spec)

    def step(_):
        if not track.done():
            track.step(agent(None, None))
            im.set_array(track.print())
        return im,

    ani = animation.FuncAnimation(fig, step, interval=50, blit=True)
    plt.show()
