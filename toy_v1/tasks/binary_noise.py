import numpy as np

from .base import Task


class BinaryNoiseTask(Task):
    """Bernoulli noise vector, independently resampled every timestep by default.

    Each element independently resamples from Bernoulli(prob) with
    probability `resample_prob` at every timestep; elements that don't
    resample hold their previous value. With the default resample_prob=1,
    every element is redrawn every timestep, giving i.i.d. Bernoulli(prob)
    noise per timestep.
    """

    def __init__(self, vector_length, prob=0.5, resample_prob=1.0):
        super().__init__(vector_length)
        self.prob = prob
        self.resample_prob = resample_prob

    def _sample_fill(self):
        return float(self._rng.binomial(1, self.prob))

    def _advance(self):
        if self._current is None:
            self._current = self._rng.binomial(
                1, self.prob, size=self.vector_length).astype(np.float64)
            return self._current
        resample_mask = self._rng.random(self.vector_length) < self.resample_prob
        resampled = self._rng.binomial(1, self.prob, size=self.vector_length).astype(np.float64)
        self._current = np.where(resample_mask, resampled, self._current)
        return self._current
