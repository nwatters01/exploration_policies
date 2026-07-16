import numpy as np

from .base import Task

# Test conditions, applied partway through a trial (see ``switch_step``).
STOP = "stop"       # freeze the stimulus (drift halts, vector held static)
SWITCH = "switch"   # reverse the drift direction (rightward -> leftward)
NOISE = "noise"     # replace drift with fresh Bernoulli noise each step
CONDITIONS = (STOP, SWITCH, NOISE)


class DriftTask(Task):
    """Vector that drifts to the right every timestep.

    Each timestep the rightmost element is dropped, all remaining elements shift
    one position to the right, and a fresh Bernoulli(prob) sample enters on the
    left.

    Optionally a *test condition* takes over at ``switch_step`` (typically halfway
    through the trial):

      - ``stop``:   the stimulus freezes (drift halts, vector held static).
      - ``switch``: the drift direction reverses (rightward -> leftward).
      - ``noise``:  drift is replaced by fresh Bernoulli(prob) noise each step.

    With ``condition=None`` (or ``switch_step=None``) the vector drifts rightward
    for the whole trial.
    """

    def __init__(self, vector_length, prob=0.5, condition=None, switch_step=None):
        super().__init__(vector_length)
        if condition is not None and condition not in CONDITIONS:
            raise ValueError(f"unknown condition: {condition!r}")
        self.prob = prob
        self.condition = condition
        self.switch_step = switch_step
        self._t = 0

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._t = 0

    def _sample_vector(self):
        return self._rng.binomial(1, self.prob, size=self.vector_length).astype(np.float64)

    def _drift(self, direction):
        new_element = float(self._rng.binomial(1, self.prob))
        if direction == "right":
            self._current = np.concatenate(([new_element], self._current[:-1]))
        else:  # "left": drop the leftmost element, shift left, new element on the right
            self._current = np.concatenate((self._current[1:], [new_element]))

    def _condition_active(self):
        return (self.condition is not None and self.switch_step is not None
                and self._t >= self.switch_step)

    def _advance(self):
        if self._current is None:
            self._current = self._sample_vector()
            self._t += 1
            return self._current

        if self._condition_active():
            if self.condition == STOP:
                pass  # hold the vector static
            elif self.condition == SWITCH:
                self._drift("left")
            elif self.condition == NOISE:
                self._current = self._sample_vector()
        else:
            self._drift("right")

        self._t += 1
        return self._current
