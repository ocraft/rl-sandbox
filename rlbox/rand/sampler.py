import itertools

import numpy as np


SAMPLER_CACHE = 10000


def cache_gen(source):
    values = source()
    while True:
        for value in values:
            yield value
        values = source()


class Sampler:
    """Provides precomputed random samples of various distribution."""

    randn_gen = cache_gen(lambda: np.random.standard_normal(SAMPLER_CACHE))
    rand_gen = cache_gen(lambda: np.random.random(SAMPLER_CACHE))

    @classmethod
    def standard_normal(cls, size=1):
        return list(itertools.islice(cls.randn_gen, size))

    @classmethod
    def randn(cls):
        return next(cls.randn_gen)

    @classmethod
    def rand(cls):
        return next(cls.rand_gen)

    @classmethod
    def rint(cls, max_exclusive):
        return np.random.randint(max_exclusive)
