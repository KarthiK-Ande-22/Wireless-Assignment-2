"""Monte Carlo estimation of the SER (Algorithm 1 of the assignment)."""
import numpy as np

from .combining import no_combining
from .detector import ml_detect


def db_to_linear(snr_db):
    """SNR = 10^(SNR_dB / 10)."""
    return 10 ** (np.asarray(snr_db, dtype=float) / 10)


def count_symbol_errors(i_hat, i):
    """Number of symbol errors, found by comparing indices."""
    return int(np.count_nonzero(i_hat != i))


def simulate_receivers(constellation, snr_db, channel, receivers, n_symbols, rng):
    """SER of one or more receivers at every SNR point.

    All receivers process the same symbols, channel and noise.
    Returns {receiver name: (SER, error count)}.
    """
    n_errors = {name: np.zeros(len(snr_db), dtype=int) for name in receivers}
    for k, snr in enumerate(db_to_linear(snr_db)):
        i = rng.integers(len(constellation), size=n_symbols)   # transmitted indices
        x = constellation[i]
        h, y = channel(x, snr, rng)
        for name, combine in receivers.items():
            r, g = combine(y, h)
            i_hat = ml_detect(r, constellation, g)
            n_errors[name][k] = count_symbol_errors(i_hat, i)
    return {name: (n / n_symbols, n) for name, n in n_errors.items()}


def simulate_ser(constellation, snr_db, channel, n_symbols, rng):
    """SER of the single-antenna receiver. Returns (SER, error count)."""
    receivers = {"ML": no_combining}
    return simulate_receivers(constellation, snr_db, channel, receivers, n_symbols, rng)["ML"]
