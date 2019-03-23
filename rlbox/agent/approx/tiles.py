"""Based on: http://incompleteideas.net/tiles/tiles3.html """

from itertools import zip_longest
from math import floor


class IHT:
    """Structure to handle collisions."""

    def __init__(self, size):
        self.size = size
        self.overfull = 0
        self.dict = {}

    def count(self):
        return len(self.dict)

    def fullp(self):
        return len(self.dict) >= self.size

    def getindex(self, obj, readonly=False):
        d = self.dict
        if obj in d:
            return d[obj]
        elif readonly:
            return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfull == 0:
                print('IHT full, starting to allow collisions')
            self.overfull += 1
            return hash(obj) % self.size
        else:
            d[obj] = count
            return count


def hashcoords(coordinates, m, readonly=False):
    if type(m) == IHT:
        return m.getindex(tuple(coordinates), readonly)
    if type(m) == int:
        return hash(tuple(coordinates)) % m
    if m is None:
        return coordinates


def tiles(iht_or_size, num_of_tilings, floats, ints=[], readonly=False):
    """Returns num-tilings tile indices corresponding to the floats and ints."""
    qfloats = [floor(f * num_of_tilings) for f in floats]
    t = []
    for tiling in range(num_of_tilings):
        tiling_x2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append((q + b) // num_of_tilings)
            b += tiling_x2
        coords.extend(ints)
        t.append(hashcoords(coords, iht_or_size, readonly))
    return t


def tiles_wrap(iht_or_size, num_of_tilings, floats, wrap_widths, ints=[],
               readonly=False):
    """Returns num-tilings tile indices corresponding to the floats and ints,
    wrapping some floats."""
    qfloats = [floor(f * num_of_tilings) for f in floats]
    tiles = []
    for tiling in range(num_of_tilings):
        tiling_x2 = tiling * 2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrap_widths):
            c = (q + b % num_of_tilings) // num_of_tilings
            coords.append(c % width if width else c)
            b += tiling_x2
        coords.extend(ints)
        tiles.append(hashcoords(coords, iht_or_size, readonly))
    return tiles
