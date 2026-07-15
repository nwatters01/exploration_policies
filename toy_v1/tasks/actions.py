"""Action space shared by all tasks.

There are three actions: do-nothing, shift-left and shift-right. ``left`` moves
the vector's contents one step to the left (the leftmost element is cut off and a
fresh element fills in on the right); ``right`` is the mirror image. Do-nothing
leaves the vector untouched.

Actions are encoded for the models as a one-hot code over {left, right}, so
do-nothing is the all-zeros vector: none -> [0, 0], left -> [1, 0], right -> [0, 1].
"""

import numpy as np

NONE = 0
LEFT = 1
RIGHT = 2

ACTIONS = (NONE, LEFT, RIGHT)
ACTION_NAMES = {NONE: "none", LEFT: "left", RIGHT: "right"}

# One-hot dimensionality: do-nothing is all zeros, so only left/right get a slot.
ACTION_DIM = 2


def apply_shift(vector, action, fill):
    """Return ``vector`` shifted by ``action``, filling the exposed cell with ``fill``.

    ``left`` drops ``vector[0]`` and appends ``fill`` on the right; ``right`` drops
    ``vector[-1]`` and prepends ``fill`` on the left; ``none`` returns a copy.
    """
    if action == NONE:
        return vector.copy()
    if action == LEFT:
        return np.concatenate([vector[1:], [fill]])
    if action == RIGHT:
        return np.concatenate([[fill], vector[:-1]])
    raise ValueError(f"unknown action: {action}")


def one_hot(actions):
    """One-hot encode an action or a 1-D array of actions over {left, right}.

    A scalar returns shape (ACTION_DIM,); a length-T array returns (T, ACTION_DIM).
    Do-nothing maps to all zeros.
    """
    actions = np.asarray(actions)
    flat = np.atleast_1d(actions)
    codes = np.zeros((flat.shape[0], ACTION_DIM), dtype=np.float64)
    codes[flat == LEFT, 0] = 1.0
    codes[flat == RIGHT, 1] = 1.0
    return codes[0] if actions.ndim == 0 else codes


def feedback(actions):
    """One-hot feedback aligned to the input it belongs to.

    The action executed at step t modifies input(t + 1), so the feedback attached
    to input(t) is the one-hot of action(t - 1). The first step has no preceding
    action, so its feedback is all zeros. Input is a length-T array of actions;
    output is (T, ACTION_DIM).
    """
    codes = one_hot(actions)
    fb = np.zeros_like(codes)
    fb[1:] = codes[:-1]
    return fb
