"""Channel models of G.2 and G.3: y = h x + w, with w ~ CN(0, 1/SNR).

Since E[|x|^2] = 1, the noise variance 1/SNR sets the (average) SNR. A fresh,
independent channel and noise sample is drawn for every transmitted symbol.
Each channel returns (h, y) so the receiver can use h for coherent detection.
"""
import numpy as np


def complex_gaussian(shape, variance, rng):
    """I.i.d. CN(0, variance) samples: sqrt(variance / 2) * (g_I + j g_Q)."""
    return np.sqrt(variance / 2) * (rng.standard_normal(shape)
                                    + 1j * rng.standard_normal(shape))


def awgn_channel(x, snr, rng):
    """AWGN channel: h = 1, y = x + w."""
    h = np.ones_like(x)
    w = complex_gaussian(x.shape, 1 / snr, rng)
    return h, h * x + w


def rayleigh_channel(x, snr, rng):
    """Flat Rayleigh fading channel: h ~ CN(0, 1), so E[|h|^2] = 1."""
    h = complex_gaussian(x.shape, 1.0, rng)
    w = complex_gaussian(x.shape, 1 / snr, rng)
    return h, h * x + w


def simo_rayleigh_channel(x, snr, rng, n_rx):
    """1 x L SIMO Rayleigh channel [2, Sec. 3]: y_l = h_l x + w_l, l = 1, ..., L.

    h_l ~ CN(0, 1) i.i.d. across antennas and symbols, w_l ~ CN(0, 1/SNR), so SNR
    is the average SNR per receive antenna. Returns h and y of shape (N, L).
    """
    shape = (x.size, n_rx)                          # one row per symbol, one column per antenna
    h = complex_gaussian(shape, 1.0, rng)           # independent fading on every branch
    w = complex_gaussian(shape, 1 / snr, rng)       # independent noise on every branch
    return h, h * x[:, np.newaxis] + w              # the same symbol x reaches all L antennas


def miso_channel(s, h, snr, rng):
    """n_tx x 1 MISO channel [2, Sec. 4]: y = sum_l h_l s_l + w, w ~ CN(0, 1/SNR).

    s has shape (n_blocks, block_len, n_tx): the signal sent from each antenna in
    each channel use of a block. h has shape (n_blocks, n_tx) and stays constant
    over the block_len channel uses of a block. The noise variance 1/SNR refers to
    a total transmitted energy of one per channel use. Returns y, shape (n_blocks, block_len).
    """
    w = complex_gaussian(s.shape[:-1], 1 / snr, rng)          # one noise sample per channel use
    return np.sum(h[:, np.newaxis, :] * s, axis=-1) + w       # antenna signals add up at the receiver
