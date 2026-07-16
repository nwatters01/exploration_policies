from abc import ABC, abstractmethod

import numpy as np


class Task(ABC):
    """Base class for tasks that produce a timeseries of vectors.

    Each task maintains a current observation vector in ``self._current`` that is
    advanced one step at a time by ``_advance`` (the task's intrinsic dynamics).
    """

    def __init__(self, vector_length):
        self.vector_length = vector_length
        self._rng = np.random.default_rng()
        self._current = None

    def reset(self, seed=None):
        """Reset internal state, optionally reseeding the RNG."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._current = None

    @abstractmethod
    def _advance(self):
        """Advance intrinsic dynamics one step, updating and returning ``self._current``."""

    def step(self):
        """Advance one step, returning a vector of shape (vector_length,)."""
        self._advance()
        return self._current.copy()

    def generate(self, num_steps, seed=None):
        """Generate a full timeseries, shape (num_steps, vector_length)."""
        self.reset(seed=seed)
        return np.stack([self.step() for _ in range(num_steps)], axis=0)
