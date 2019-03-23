import numpy as np

from rlbox.testbed.sink import Sink


class TestPerfData:
    def test_created_shared_mem_from_config(self):
        with Sink({
                'rewards': (2000, 1000, 'i'),
                'actions': (2000, 1500, 'd')
        }, '.dump/tmp') as data:
            assert data.to_share() is not None
            assert 'rewards' in data.get()
            assert 'actions' in data.get()

    def test_collects_data(self):
        with Sink({'rewards': (2, 2, 'd')}, '.dump/tmp') as data:
            rewards = np.array([1.0, 2.0])

            data.dump('rewards', 1, rewards)

            mem = data.get()
            assert np.array_equal(mem['rewards'][1], rewards)
