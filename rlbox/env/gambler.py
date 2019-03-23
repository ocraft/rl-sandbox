import numpy as np

from .mdp import Mdp


class Gambler(Mdp):
    def __init__(self, ph, stake):
        Mdp.__init__(self, 0)
        self.ph = ph
        self.stake = stake
        self.S = np.arange(0, stake + 1)
        self.A = np.arange(1, stake)

    def prepare_model(self):
        ns = len(self.S)
        na = len(self.A)

        # transition matrix
        t = np.zeros((ns, ns, na), dtype=np.float64)

        # reward matrix
        r = np.zeros(t.shape, dtype=np.int32)

        for s in self.S[1:-1]:
            for ai, a in enumerate(np.arange(1, min(s, self.stake - s) + 1)):
                next_s_lose = s - a
                next_s_win = s + a

                t[s, next_s_lose, ai] += 1 - self.ph
                t[s, next_s_win, ai] += self.ph

                r[s, next_s_lose, ai] = 0.0
                r[s, next_s_win, ai] = 1.0 if next_s_win == self.stake else 0.0

        self.P = t
        self.R = r
