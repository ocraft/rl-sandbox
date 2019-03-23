import numpy as np
import pytest

from rlbox.env.narmedbandit import Bandit, NonstationaryBandit


class TestBandit:
    def test_prepares_qstart_means_for_each_arm(self):
        arms_count = 5
        assert len(Bandit(arms_count).qstar_means) == arms_count

    def test_provides_best_action_value(self):
        bandit = Bandit(5)
        assert bandit.best_action == np.argmax(bandit.qstar_means)

    def test_provides_reward_for_pulling_arm(self):
        bandit = Bandit(5)
        assert bandit.pull(3) is not None

    def test_raises_error_when_invalid_arm_is_pulled(self):
        with pytest.raises(IndexError):
            Bandit(1).pull(2)


class TestNonstationaryBandit:
    def test_prepares_equal_qstar_means_at_start(self):
        bandit = NonstationaryBandit(5)
        assert np.all(bandit.qstar_means == bandit.qstar_means[0])

    def test_updates_qstar_means_after_each_pull(self):
        bandit = NonstationaryBandit(5)
        old_qstar_means = bandit.qstar_means

        assert bandit.pull(3) is not None
        assert np.all(bandit.qstar_means != old_qstar_means)

    def test_updates_best_action_for_each_pull(self):
        bandit = NonstationaryBandit(5)
        bandit.best_action = -1
        old_best_action = bandit.best_action

        assert bandit.pull(3) is not None
        assert bandit.best_action != old_best_action
