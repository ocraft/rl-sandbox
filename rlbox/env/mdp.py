from abc import ABC, abstractmethod


class Mdp(ABC):
    """Markov Decision Process"""

    def __init__(self, S0):
        self.S0 = S0
        self._S = None
        self._A = None
        self._P = None
        self._R = None

    @property
    def S(self):
        """vector: a finite set of states"""
        return self._S

    @property
    def A(self):
        """vector: enumeration of all possible actions"""
        return self._A

    @property
    def P(self):
        """matrix: P(s, a, s') - transition probability matrix"""
        return self._P

    @property
    def R(self):
        """matrix: R(s, a, s') - matrix with reward for each transition"""
        return self._R

    @S.setter
    def S(self, value):
        self._S = value

    @A.setter
    def A(self, value):
        self._A = value

    @P.setter
    def P(self, value):
        self._P = value

    @R.setter
    def R(self, value):
        self._R = value

    @abstractmethod
    def prepare_model(self):
        """Generates MDP model."""
