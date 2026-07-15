from abc import ABC, abstractmethod

import numpy as np


class Task(ABC):
    """Base class for tasks that produce a timeseries of vectors."""

    def __init__(self, vector_length):
        self.vector_length = vector_length
        self._rng = np.random.default_rng()

    def reset(self, seed=None):
        """Reset internal state, optionally reseeding the RNG."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @abstractmethod
    def step(self):
        """Advance one timestep, returning a vector of shape (vector_length,)."""

    def generate(self, num_steps, seed=None):
        """Generate a full timeseries, shape (num_steps, vector_length)."""
        self.reset(seed=seed)
        return np.stack([self.step() for _ in range(num_steps)], axis=0)
