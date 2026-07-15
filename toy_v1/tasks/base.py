from abc import ABC, abstractmethod

import numpy as np

from .actions import NONE, apply_shift


class Task(ABC):
    """Base class for tasks that produce a timeseries of vectors.

    Each task maintains a current observation vector in ``self._current`` that is
    advanced one step at a time by ``_advance`` (the task's intrinsic dynamics).
    On top of that, ``step`` can apply an action -- ``left``/``right`` shift the
    vector, cutting off one end and filling the exposed cell with a fresh sample
    from ``_sample_fill`` (do-nothing leaves it untouched). The shift is applied
    to ``self._current``, so for tasks whose dynamics carry that state forward
    (e.g. drift) the effect of an action persists into subsequent steps.
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

    @abstractmethod
    def _sample_fill(self):
        """Return a fresh scalar for a cell newly exposed by a left/right shift."""

    def step(self, action=NONE):
        """Advance one step under ``action``, returning a vector of shape (vector_length,)."""
        self._advance()
        if action != NONE:
            self._current = apply_shift(self._current, action, self._sample_fill())
        return self._current.copy()

    def generate(self, num_steps, seed=None):
        """Generate a full action-free timeseries, shape (num_steps, vector_length)."""
        self.reset(seed=seed)
        return np.stack([self.step() for _ in range(num_steps)], axis=0)
