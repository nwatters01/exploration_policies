"""Drive a policy over a task to produce an observation/action rollout."""

import numpy as np

from .actions import NONE


def rollout(task, policy, num_steps, seed=None):
    """Roll a policy out over a task for ``num_steps`` steps.

    The action chosen at step t modifies the observation at step t + 1, so the
    returned arrays are aligned as: ``actions[t]`` was taken having seen
    ``observations[t]``, and (for t < num_steps - 1) produced
    ``observations[t + 1]``. The final action's resulting observation falls one
    step past the end of the window, matching how the model predicts one step
    past the sequence.

    Args:
        task: a ``Task`` instance.
        policy: a ``Policy`` instance, or ``None`` for an all do-nothing rollout.
        num_steps: length T of the returned sequences.
        seed: optional seed; the task and policy are seeded deterministically
            from it so a given seed reproduces the whole rollout.

    Returns:
        observations: float array (num_steps, vector_length).
        actions: int array (num_steps,) of action ids.
    """
    task.reset(seed=seed)
    if policy is not None:
        policy.reset(seed=None if seed is None else seed + 1)

    observations = []
    actions = []
    obs = task.step(NONE)  # input(0)
    for _ in range(num_steps):
        action = NONE if policy is None else policy.act(obs)
        observations.append(obs)
        actions.append(action)
        obs = task.step(action)  # input(t + 1), modified by this action

    return np.stack(observations, axis=0), np.array(actions, dtype=np.int64)
