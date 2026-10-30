"""Theoretical symbol error probabilities (SNR in linear scale)."""
import numpy as np
from scipy.special import erfc


def qfunc(a):
    """Gaussian tail function Q(a) = 0.5 erfc(a / sqrt(2))."""
    return 0.5 * erfc(np.asarray(a) / np.sqrt(2))


def ser_bpsk_awgn(snr):
    """BPSK in AWGN: Q(sqrt(2 SNR))."""
    return qfunc(np.sqrt(2 * snr))


def ser_qam4_awgn(snr):
    """4-QAM in AWGN: 2 Q(sqrt(SNR)) - Q^2(sqrt(SNR))."""
    q = qfunc(np.sqrt(snr))   # error probability of one component (I or Q)
    return 2 * q - q ** 2


def ser_bpsk_rayleigh(snr):
    """BPSK in Rayleigh fading: (1/2) (1 - sqrt(SNR / (1 + SNR)))."""
    return 0.5 * (1 - np.sqrt(snr / (1 + snr)))


def ser_qam4_rayleigh(snr):
    """4-QAM in Rayleigh fading, approximation of the assignment table (twice BPSK at SNR/2)."""
    return 2 * ser_bpsk_rayleigh(snr / 2)


def ser_qam4_rayleigh_exact(snr):
    """4-QAM in Rayleigh fading, exact: also counts the case where I and Q are both wrong."""
    mu = np.sqrt((snr / 2) / (1 + snr / 2))
    both_in_error = 0.25 * (1 - (4 / np.pi) * mu * np.arctan(1 / mu))
    return (1 - mu) - both_in_error


def ser_qam4_fading_mgf(snr, mgf, n_nodes=256):
    """Exact 4-QAM SER for a random channel gain gamma with MGF M(s) = E[exp(-s gamma)].

    Craig's formula:
    Pe = (2/pi) int_0^{pi/2} M(SNR / (2 sin^2 t)) dt - (1/pi) int_0^{pi/4} M(SNR / (2 sin^2 t)) dt,
    computed with Gauss-Legendre quadrature.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    snr = np.asarray(snr, dtype=float)[..., np.newaxis]

    def craig_integral(upper):
        t = upper / 2 * (nodes + 1)   # quadrature nodes moved from [-1, 1] to [0, upper]
        return upper / 2 * np.sum(weights * mgf(snr / (2 * np.sin(t) ** 2)), axis=-1)

    return (2 * craig_integral(np.pi / 2) - craig_integral(np.pi / 4)) / np.pi


def ser_qam4_mrc(snr, n_rx):
    """4-QAM with L-branch MRC (or beamforming): gamma = ||h||^2, M(s) = (1 + s)^-L."""
    return ser_qam4_fading_mgf(snr, lambda s: (1 + s) ** -float(n_rx))


def ser_qam4_sc(snr, n_rx):
    """4-QAM with L-branch SC: gamma = max |h_l|^2, M(s) = prod_k k / (k + s)."""
    return ser_qam4_fading_mgf(snr, lambda s: np.prod([k / (k + s) for k in range(1, n_rx + 1)], axis=0))


def ser_qam4_alamouti(snr):
    """4-QAM with Alamouti: gamma = ||h||^2 / 2, M(s) = (1 + s/2)^-2."""
    return ser_qam4_fading_mgf(snr, lambda s: (1 + s / 2) ** -2.0)
