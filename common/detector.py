"""Maximum-likelihood detector (G.4; lecture notes [1, Secs. 4.1 and 5.1])."""
import numpy as np


def ml_detect(y, constellation, h=1.0):
    """Coherent ML detection of every received sample, for any constellation.

    For equiprobable symbols in circularly-symmetric Gaussian noise, ML reduces
    to choosing the point of the scaled constellation {h u_1, ..., h u_M} that
    is closest to y:

        i_hat = argmin_m |y - h u_m|^2.

    `h` is the channel known at the receiver: a scalar (h = 1 for AWGN) or an
    array of the same shape as `y`. The full N x M distance array is formed at
    once and the minimising index along the constellation axis is returned.
    """
    y = np.asarray(y)[..., np.newaxis]               # (N, 1)
    h = np.asarray(h)[..., np.newaxis]               # (N, 1), or (1,) if scalar
    distances = np.abs(y - h * constellation) ** 2   # (N, M)
    return np.argmin(distances, axis=-1)
