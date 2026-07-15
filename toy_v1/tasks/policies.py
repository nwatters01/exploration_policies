"""Policies that choose actions for a task.

A policy maps the current observation to an action each timestep. This module
provides a small base class so more policies can be added later; ``rollout``
(see ``tasks.rollout``) is what actually drives a policy over a task.
"""

from abc import ABC, abstractmethod

import numpy as np

from .actions import LEFT, NONE, RIGHT


class Policy(ABC):
    """Base class: maps an observation to an action id each timestep."""

    def __init__(self):
        self._rng = np.random.default_rng()

    def reset(self, seed=None):
        """Reset internal state, optionally reseeding the RNG."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, observation):
        """Return an action id (NONE / LEFT / RIGHT) for the given observation."""


class RandomShiftPolicy(Policy):
    """Act randomly, independent of the observation.

    With probability ``action_probability`` a shift is taken (left or right,
    each equally likely); otherwise the do-nothing action is chosen.
    """

    def __init__(self, action_probability=0.5):
        super().__init__()
        if not 0.0 <= action_probability <= 1.0:
            raise ValueError(f"action_probability must be in [0, 1], got {action_probability}")
        self.action_probability = action_probability

    def act(self, observation):
        if self._rng.random() < self.action_probability:
            return LEFT if self._rng.random() < 0.5 else RIGHT
        return NONE
