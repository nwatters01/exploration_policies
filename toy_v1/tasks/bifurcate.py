import numpy as np

from .base import Task


class BifurcateTask(Task):
    """Ones breathe in from the edges, then zeros return one of two ways.

    A cycle proceeds:

    * **Grow:** starting from all zeros, a ``1`` is added at each edge every step,
      so the ones encroach inward until they meet (all ones).
    * **Shrink (the bifurcation):** once the ones meet, zeros return in one of two
      equally likely ways:
        - from the **middle**, growing outward, so the ones retreat to the ends;
        - from the **ends**, growing inward, so the ones shrink in the middle.
      Zeros grow until no ones remain (all zeros), at which point the cycle
      starts over with the ones encroaching from the ends again.

    Which of the two shrink modes happens is the genuinely unpredictable event.

    Test mode: once the ones meet, the zeros instead encroach **double-fast from
    one of the ends** (left or right, chosen at random) -- an out-of-distribution
    return never shown in normal trials. (As with the other tasks the test change
    takes effect from the trial midpoint onward, so shrink phases that begin after
    the midpoint use the fast one-ended return.)
    """

    def __init__(self, vector_length):
        super().__init__(vector_length)
        self._phase = "grow"   # 'grow' (ones from edges) or 'shrink' (zeros return)
        self._p = 0            # step counter within the current phase
        self._mode = "middle"  # active shrink mode

    def reset(self, seed=None):
        super().reset(seed=seed)
        self._phase = "grow"
        self._p = 0
        self._mode = "middle"

    def _sample_fill(self):
        return float(self._rng.binomial(1, 0.5))

    def _choose_shrink_mode(self):
        if self._test_active():
            return "left" if self._rng.random() < 0.5 else "right"
        return "middle" if self._rng.random() < 0.5 else "ends"

    def _shrink_frame(self, q, mode):
        """Return (vector, done) for step ``q`` of a shrink phase in ``mode``."""
        L = self.vector_length
        v = np.zeros(L)
        if mode == "middle":                 # zeros from the middle; ones retreat to the ends
            m = L // 2 - q                    # ones remaining at each end
            if m <= 0:
                return v, True
            v[:m] = 1.0
            v[L - m:] = 1.0
            return v, False
        if mode == "ends":                   # zeros from both ends; ones shrink in the middle
            w = L - 2 * q                     # width of the central ones block
            if w <= 0:
                return v, True
            start = (L - w) // 2
            v[start:start + w] = 1.0
            return v, False
        # test: zeros encroach double-fast from a single end
        z = 2 * q                            # number of zeros eaten from that end
        if z >= L:
            return v, True
        if mode == "left":
            v[z:] = 1.0
        else:  # "right"
            v[:L - z] = 1.0
        return v, False

    def _advance(self):
        L = self.vector_length
        if self._current is None:
            self._phase = "grow"
            self._p = 0
            self._current = np.zeros(L)  # start all zeros
            return self._current

        if self._phase == "grow":
            self._p += 1
            p = self._p
            if 2 * p >= L:                   # the ones meet -> all ones, then start shrinking
                self._current = np.ones(L)
                self._phase = "shrink"
                self._p = 0
                self._mode = self._choose_shrink_mode()
            else:
                v = np.zeros(L)
                v[:p] = 1.0
                v[L - p:] = 1.0
                self._current = v
            return self._current

        # shrink phase
        self._p += 1
        v, done = self._shrink_frame(self._p, self._mode)
        self._current = v
        if done:                             # no ones left -> restart the growth ramp
            self._phase = "grow"
            self._p = 0
        return self._current
