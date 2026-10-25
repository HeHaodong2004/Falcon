"""Shared observation-horizon contract; motion is not observation count."""
import numpy as np


def observation_probabilities(probabilities, *, no_future_observations):
    """Return owned probabilities, forcing no update only for an empty horizon.

    A one-pose trajectory has no later observation. A stationary path with
    later timestamps can reveal cells and must retain its model output.
    """
    p = np.array(probabilities, dtype=np.float32, copy=True)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError('Expected UNKNOWN/FREE/BLOCKED probabilities')
    if no_future_observations:
        p.fill(0)
        p[:, 0] = 1
    return p
