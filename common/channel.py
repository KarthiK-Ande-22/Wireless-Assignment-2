"""Channel models: y = h x + w with w ~ CN(0, 1/SNR). Each returns (h, y)."""
import numpy as np


def complex_gaussian(shape, variance, rng):
    """Independent CN(0, variance) samples."""
    return np.sqrt(variance / 2) * (rng.standard_normal(shape)
                                    + 1j * rng.standard_normal(shape))


def awgn_channel(x, snr, rng):
    """AWGN channel (h = 1)."""
    h = np.ones_like(x)
    w = complex_gaussian(x.shape, 1 / snr, rng)
    return h, h * x + w


def rayleigh_channel(x, snr, rng):
    """Rayleigh fading: a new h ~ CN(0, 1) for every symbol."""
    h = complex_gaussian(x.shape, 1.0, rng)
    w = complex_gaussian(x.shape, 1 / snr, rng)
    return h, h * x + w


def simo_rayleigh_channel(x, snr, rng, n_rx):
    """1 x L channel: the same x reaches L antennas, each with its own h_l and w_l.

    h and y have shape (N, L).
    """
    shape = (x.size, n_rx)
    h = complex_gaussian(shape, 1.0, rng)
    w = complex_gaussian(shape, 1 / snr, rng)
    return h, h * x[:, np.newaxis] + w


def miso_channel(s, h, snr, rng):
    """2 x 1 channel: y = h_1 s_1 + h_2 s_2 + w in every channel use.

    s has shape (blocks, channel uses per block, antennas); h stays fixed within a block.
    """
    w = complex_gaussian(s.shape[:-1], 1 / snr, rng)
    return np.sum(h[:, np.newaxis, :] * s, axis=-1) + w
