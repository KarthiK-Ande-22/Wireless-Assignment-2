"""Monte Carlo estimation of the symbol error rate (G.5, Algorithm 1)."""
import numpy as np

from .combining import no_combining
from .detector import ml_detect


def db_to_linear(snr_db):
    """SNR = 10^(SNR_dB / 10)."""
    return 10 ** (np.asarray(snr_db, dtype=float) / 10)


def count_symbol_errors(i_hat, i):
    """Number of symbol errors, found by comparing indices rather than complex values."""
    return int(np.count_nonzero(i_hat != i))


def simulate_receivers(constellation, snr_db, channel, receivers, n_symbols, rng):
    """Estimate the SER of one or more receivers at every SNR point.

    For each SNR: draw `n_symbols` equiprobable indices i, send x = u_i through
    `channel(x, snr, rng) -> (h, y)`, and let every receiver process the *same*
    realisations. A receiver `combine(y, h) -> (r, g)` (see common/combining.py)
    reduces the observation to a scalar channel r = g x + noise, which is then
    detected by the ML detector. Returns {name: (SER estimates n_err / N, counts n_err)}.
    """
    n_errors = {name: np.zeros(len(snr_db), dtype=int) for name in receivers}
    for k, snr in enumerate(db_to_linear(snr_db)):
        i = rng.integers(len(constellation), size=n_symbols)   # ground truth
        x = constellation[i]                                   # transmitter
        h, y = channel(x, snr, rng)                            # channel
        for name, combine in receivers.items():
            r, g = combine(y, h)                               # combiner
            i_hat = ml_detect(r, constellation, g)             # detector
            n_errors[name][k] = count_symbol_errors(i_hat, i)
    return {name: (n / n_symbols, n) for name, n in n_errors.items()}


def simulate_ser(constellation, snr_db, channel, n_symbols, rng):
    """SER of the single-antenna coherent ML receiver. Returns (SER estimates, counts)."""
    receivers = {"ML": no_combining}
    return simulate_receivers(constellation, snr_db, channel, receivers, n_symbols, rng)["ML"]
