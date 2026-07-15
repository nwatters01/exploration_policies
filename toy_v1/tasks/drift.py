import numpy as np

from .base import Task


class DriftTask(Task):
    """Vector that drifts to the right every timestep.

    Each timestep the rightmost element is dropped, all remaining elements
    shift one position to the right, and a fresh Bernoulli(prob) sample
    enters on the left.

    Because the drift carries ``self._current`` forward, a ``left`` action can
    counteract the intrinsic rightward drift (and ``right`` compounds it).
    """

    def __init__(self, vector_length, prob=0.5):
        super().__init__(vector_length)
        self.prob = prob

    def _sample_fill(self):
        return float(self._rng.binomial(1, self.prob))

    def _advance(self):
        if self._current is None:
            self._current = self._rng.binomial(
                1, self.prob, size=self.vector_length).astype(np.float64)
            return self._current
        new_element = float(self._rng.binomial(1, self.prob))
        self._current = np.concatenate(([new_element], self._current[:-1]))
        return self._current
