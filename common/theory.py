"""Theoretical symbol error probabilities. All SNR arguments are linear."""
import numpy as np
from scipy.special import erfc


def qfunc(a):
    """Gaussian tail function Q(a) = Pr[N(0, 1) > a] = 0.5 erfc(a / sqrt(2))."""
    return 0.5 * erfc(np.asarray(a) / np.sqrt(2))


def ser_bpsk_awgn(snr):
    """BPSK in AWGN [1, Sec. 4.3]: Pe = Q(sqrt(2 SNR))."""
    return qfunc(np.sqrt(2 * snr))


def ser_qam4_awgn(snr):
    """4-QAM in AWGN: Pe = 2 Q(sqrt(SNR)) - Q^2(sqrt(SNR)).

    I and Q are detected independently, each as BPSK carrying half the energy
    (per-component error q = Q(sqrt(SNR))), and the symbol is correct only if
    both components are: Pe = 1 - (1 - q)^2.
    """
    q = qfunc(np.sqrt(snr))
    return 2 * q - q ** 2


def ser_bpsk_rayleigh(snr):
    """BPSK in Rayleigh fading [1, Sec. 5.2]: Pe = (1/2) (1 - sqrt(SNR / (1 + SNR)))."""
    return 0.5 * (1 - np.sqrt(snr / (1 + snr)))


def ser_qam4_rayleigh(snr):
    """4-QAM in Rayleigh fading, approximation of the assignment table.

    Twice the BPSK result at SNR/2. It drops the event that both I and Q are
    in error, so it is an upper bound on the true SER.
    """
    return 2 * ser_bpsk_rayleigh(snr / 2)


def ser_qam4_rayleigh_exact(snr):
    """4-QAM in Rayleigh fading, exact: Pe = E[2 Q(sqrt(SNR |h|^2)) - Q^2(sqrt(SNR |h|^2))].

    With mu = sqrt((SNR/2) / (1 + SNR/2)), the first term averages to 1 - mu, and
    the joint I/Q error term (via Craig's form of Q^2) to
    (1/4) (1 - (4/pi) mu arctan(1/mu)). Used only to explain the gap between
    simulation and the approximation above.
    """
    mu = np.sqrt((snr / 2) / (1 + snr / 2))
    both_in_error = 0.25 * (1 - (4 / np.pi) * mu * np.arctan(1 / mu))
    return (1 - mu) - both_in_error


def ser_qam4_fading_mgf(snr, mgf, n_nodes=256):
    """Exact 4-QAM SER when the post-detection SNR is gamma * SNR with a random gain gamma.

    Every receiver of Problems 2 and 3 gives r = g x + n with n ~ CN(0, 1/SNR), so
    gamma = |g|^2. For a fixed gamma, Pe = 2 Q(sqrt(gamma SNR)) - Q^2(sqrt(gamma SNR)).
    Craig's forms Q(x) = (1/pi) int_0^{pi/2} exp(-x^2 / (2 sin^2 t)) dt and
    Q^2(x) = (1/pi) int_0^{pi/4} exp(-x^2 / (2 sin^2 t)) dt turn the average over
    gamma into integrals of its moment-generating function M(s) = E[exp(-s gamma)]:

        Pe = (2/pi) int_0^{pi/2} M(SNR / (2 sin^2 t)) dt - (1/pi) int_0^{pi/4} M(SNR / (2 sin^2 t)) dt,

    evaluated here with Gauss-Legendre quadrature.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)        # quadrature rule on [-1, 1]
    snr = np.asarray(snr, dtype=float)[..., np.newaxis]               # (n_snr, 1) against (n_nodes,)

    def craig_integral(upper):
        t = upper / 2 * (nodes + 1)                                   # nodes mapped to [0, upper]
        return upper / 2 * np.sum(weights * mgf(snr / (2 * np.sin(t) ** 2)), axis=-1)

    return (2 * craig_integral(np.pi / 2) - craig_integral(np.pi / 4)) / np.pi


def ser_qam4_mrc(snr, n_rx):
    """4-QAM with L-branch MRC (or L-antenna transmit beamforming), exact.

    gamma = ||h||^2 ~ Gamma(L, 1), so M(s) = (1 + s)^(-L). For L = 1 this is the
    single-antenna Rayleigh channel of Problem 1.
    """
    return ser_qam4_fading_mgf(snr, lambda s: (1 + s) ** -float(n_rx))


def ser_qam4_sc(snr, n_rx):
    """4-QAM with L-branch selection combining, exact.

    gamma = max_l |h_l|^2 has the distribution of sum_k E_k / k with E_k ~ Exp(1)
    independent, k = 1, ..., L, so M(s) = prod_k k / (k + s).
    """
    return ser_qam4_fading_mgf(snr, lambda s: np.prod([k / (k + s) for k in range(1, n_rx + 1)], axis=0))


def ser_qam4_alamouti(snr):
    """4-QAM with the 2 x 1 Alamouti code, exact.

    gamma = ||h||^2 / 2 with ||h||^2 ~ Gamma(2, 1), so M(s) = (1 + s/2)^(-2).
    """
    return ser_qam4_fading_mgf(snr, lambda s: (1 + s / 2) ** -2.0)
