"""ML (minimum-distance) detector, used by all three problems."""
import numpy as np


def ml_detect(y, constellation, h=1.0):
    """Return the index of the closest point, argmin_m |y - h u_m|^2.

    h is the known channel gain (1 for AWGN). All N samples are detected at once
    using an N x M array of distances.
    """
    y = np.asarray(y)[..., np.newaxis]
    h = np.asarray(h)[..., np.newaxis]
    distances = np.abs(y - h * constellation) ** 2
    return np.argmin(distances, axis=-1)
