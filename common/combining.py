"""Receive combiners [2, Sec. 3]: reduce an observation to an equivalent scalar channel.

Every combiner maps the received samples y and the known channel h to (r, g) with

    r = g x + n,    n ~ CN(0, 1/SNR),

which has the same form as the single-antenna channel of Problem 1. The ML
detector of G.4 is therefore reused unchanged: i_hat = argmin_m |r - g u_m|^2.
For L antennas, y and h have shape (N, L) and r, g have shape (N,).
"""
import numpy as np


def no_combining(y, h):
    """Single antenna: y = h x + w is already scalar (gain h, noise variance 1/SNR)."""
    return y, h


def mrc_combine(y, h):
    """Maximal ratio combining [2, Secs. 3.1-3.2].

    r = h^H y / ||h|| = ||h|| x + h^H w / ||h||. The effective gain is ||h|| (real,
    positive) and the noise h^H w / ||h|| is still CN(0, 1/SNR), so the
    post-combining SNR is ||h||^2 SNR = sum_l |h_l|^2 SNR.
    """
    h_norm = np.linalg.norm(h, axis=-1)                 # ||h|| = sqrt(sum_l |h_l|^2), per symbol
    r = np.sum(np.conj(h) * y, axis=-1) / h_norm        # co-phase each branch, weight by |h_l|, add
    return r, h_norm


def selection_combine(y, h):
    """Selection combining [2, Sec. 3.5].

    Only the branch l* = argmax_l |h_l|^2 is used: r = y_l* = h_l* x + w_l*. The
    effective gain is h_l* and the noise is CN(0, 1/SNR), so the post-combining
    SNR is max_l |h_l|^2 SNR.
    """
    best = np.argmax(np.abs(h) ** 2, axis=-1)           # strongest antenna for every symbol
    rows = np.arange(h.shape[0])                        # pair each symbol with its chosen antenna
    return y[rows, best], h[rows, best]                 # received sample and gain of that branch
