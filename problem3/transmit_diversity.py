"""Problem 3: transmit diversity over a 2 x 1 MISO Rayleigh channel, 4-QAM.

h_1, h_2 ~ CN(0, 1) i.i.d., known at the receiver, noise w ~ CN(0, 1/SNR). Every
scheme transmits a total energy of one per channel use, summed over the antennas,
as the single-antenna system does:
  - single antenna (reference): the fading channel of Problem 1;
  - transmit beamforming [2, Sec. 4.1]: the transmitter knows h and sends v x,
    v = h* / ||h||, so the two antenna signals add coherently;
  - Alamouti [2, Sec. 4.2]: no channel knowledge at the transmitter; two symbols
    in two channel uses, h constant within a block and independent across blocks
    (N/2 blocks, so N symbols in total).
The curves are compared with MRC for L = 2 (Problem 2). The script also measures
the average transmitted energy per channel use, the agreement with exact theory,
the high-SNR slopes, the array gains and the SNR gaps quoted in the discussion.

Run from the repository root:   python -m problem3.transmit_diversity
Outputs:                        results/q3/transmit_diversity.{csv,pdf}, results/q3/images/transmit_diversity.png
                                results/q3/transmit_diversity_theory.{csv,pdf}, results/q3/images/transmit_diversity_theory.png
                                results/q3/summary.txt
"""
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import numpy as np

from common.analysis import (agreement_lines, format_db, format_gap, high_snr_slopes, loglog_slope,
                             snr_at_ser, theory_snr_at_ser, z_score)
from common.channel import complex_gaussian, miso_channel, simo_rayleigh_channel
from common.combining import mrc_combine, no_combining
from common.constellation import qam4
from common.detector import ml_detect
from common.plotting import PALETTE, GapArrow, SimCurve, TheoryCurve, save_results
from common.simulation import count_symbol_errors, db_to_linear, simulate_receivers
from common.theory import ser_qam4_alamouti, ser_qam4_mrc, ser_qam4_rayleigh_exact

SEED = 2026
N_SYMBOLS = 10**6                                     # symbols per SNR point (N/2 Alamouti blocks)
SNR_DB = np.arange(0, 25, 2)                          # {0, 2, ..., 24} dB
SLOPE_SEGMENTS, SLOPE_MIN_ERRORS = 3, 50              # local slopes: last 3 steps, >= 50 errors
ASYMPTOTIC_DB = (60, 70)                              # SNR range of the asymptotic (exact theory) slope
GAP_TARGETS = (1e-2, 1e-3, 1e-4)                      # SER levels at which SNR gaps are read
EXACT_THEORY = {                                      # snr -> exact 4-QAM SER of each curve
    "Single antenna": ser_qam4_rayleigh_exact,        # gamma = |h|^2
    "Beamforming": partial(ser_qam4_mrc, n_rx=2),     # gamma = ||h||^2, as for MRC with L = 2
    "Alamouti": ser_qam4_alamouti,                    # gamma = ||h||^2 / 2
    "MRC L=2": partial(ser_qam4_mrc, n_rx=2),
}
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "q3"


# ---------------------------------------------------------------------------
# Transmitters: symbols x of shape (n_blocks, block_len) and channel h of shape
# (n_blocks, n_tx) -> signal s of shape (n_blocks, block_len, n_tx), i.e. what
# every antenna sends in every channel use. Only beamforming uses h (CSIT).
# ---------------------------------------------------------------------------
def single_antenna_tx(x, h):
    """Reference: x is sent unscaled from one antenna, so E|s|^2 = E|x|^2 = 1."""
    return x[..., np.newaxis]                                         # (N, 1 use, 1 antenna)


def beamforming_tx(x, h):
    """Transmit beamforming [2, Sec. 4.1]: s = v x with v = h* / ||h||.

    The conjugate phases make both antenna signals arrive in phase (h^T v = ||h||),
    and ||v|| = 1 keeps the total energy per channel use at E|x|^2 = 1.
    """
    v = np.conj(h) / np.linalg.norm(h, axis=-1, keepdims=True)        # unit-norm weights, (N, 2)
    return x[..., np.newaxis] * v[:, np.newaxis, :]                   # same symbol on both antennas, (N, 1, 2)


def alamouti_tx(x, h):
    """Alamouti code [2, Sec. 4.2]; h is ignored (no CSI at the transmitter).

    Rows are channel uses, columns are antennas:

        S = (1/sqrt(2)) [[  x1,   x2  ],
                         [ -x2*,  x1* ]]

    The factor 1/sqrt(2) splits the unit energy of every channel use over the two antennas.
    """
    x1, x2 = x[:, 0], x[:, 1]                                         # the two symbols of each block
    use1 = np.stack([x1, x2], axis=-1)                                # channel use 1: antennas send x1, x2
    use2 = np.stack([-np.conj(x2), np.conj(x1)], axis=-1)             # channel use 2: antennas send -x2*, x1*
    return np.stack([use1, use2], axis=1) / np.sqrt(2)                # (N/2, 2 uses, 2 antennas)


# ---------------------------------------------------------------------------
# Receivers: (y, h) -> (r, g), one equivalent scalar channel r = g x + n with
# n ~ CN(0, 1/SNR) per transmitted symbol, detected by the ML detector of Problem 1.
# The single-antenna receiver is `no_combining` (r = y, g = h).
# ---------------------------------------------------------------------------
def beamforming_rx(y, h):
    """y = (h^T v) x + w = ||h|| x + w: gain ||h||, noise CN(0, 1/SNR)."""
    return y, np.linalg.norm(h, axis=-1, keepdims=True)               # (N, 1) each


def alamouti_rx(y, h):
    """Alamouti combining: r_k = (||h|| / sqrt(2)) x_k + n_k, n_k ~ CN(0, 1/SNR), k = 1, 2.

    r1 = (h1* y1 + h2 y2*) / ||h||  and  r2 = (h2* y1 - h1 y2*) / ||h||. The
    orthogonal code cancels the other symbol, and n1, n2 are independent, so
    symbol-by-symbol ML detection is optimal.
    """
    h1, h2 = h[:, 0], h[:, 1]                                         # channel of each block
    y1, y2 = y[:, 0], y[:, 1]                                         # received samples of the two uses
    h_norm = np.linalg.norm(h, axis=-1)                               # ||h||
    r1 = (np.conj(h1) * y1 + h2 * np.conj(y2)) / h_norm               # x2 cancels, x1 remains
    r2 = (np.conj(h2) * y1 - h1 * np.conj(y2)) / h_norm               # x1 cancels, x2 remains
    gain = h_norm / np.sqrt(2)                                        # same effective gain for x1 and x2
    return np.stack([r1, r2], axis=-1), np.stack([gain, gain], axis=-1)


@dataclass
class MisoScheme:
    """One transmission scheme: block structure, transmitter and receiver."""
    name: str
    n_tx: int                  # transmit antennas
    block_len: int             # channel uses per block (h constant within a block)
    transmit: Callable         # (x, h) -> s
    receive: Callable          # (y, h) -> (r, g)
    scaling: str               # the scaling applied to the symbols, for the summary


SCHEMES = [
    MisoScheme("Single antenna", 1, 1, single_antenna_tx, no_combining, "s = x"),
    MisoScheme("Beamforming", 2, 1, beamforming_tx, beamforming_rx, "s_l = (h_l* / ||h||) x"),
    MisoScheme("Alamouti", 2, 2, alamouti_tx, alamouti_rx, "s_l = +-x_k(*) / sqrt(2)"),
]


@dataclass
class SchemeResult:
    """Monte Carlo output of one scheme, one entry per SNR point."""
    ser: np.ndarray            # SER estimates n_err / N
    n_errors: np.ndarray       # error counts
    energy: np.ndarray         # average |s_l|^2 per channel use, shape (n_snr, n_tx)
    gain: np.ndarray           # average effective power gain E|g|^2 (array gain)


def simulate_scheme(scheme, constellation, rng):
    """Monte Carlo SER of one scheme: Algorithm 1 with a transmitter stage.

    At every SNR it also measures the average transmitted energy per channel use
    on each antenna, and the average power gain E|g|^2 of the equivalent channel.
    """
    n_blocks = N_SYMBOLS // scheme.block_len                          # N/2 blocks for Alamouti
    n_errors = np.zeros(len(SNR_DB), dtype=int)
    energy = np.zeros((len(SNR_DB), scheme.n_tx))
    gain = np.zeros(len(SNR_DB))
    for k, snr in enumerate(db_to_linear(SNR_DB)):
        i = rng.integers(len(constellation), size=N_SYMBOLS)          # ground truth
        x = constellation[i].reshape(n_blocks, scheme.block_len)      # symbols grouped into blocks
        h = complex_gaussian((n_blocks, scheme.n_tx), 1.0, rng)       # new channel for every block
        s = scheme.transmit(x, h)                                     # antenna signals
        y = miso_channel(s, h, snr, rng)                              # received samples
        r, g = scheme.receive(y, h)                                   # equivalent scalar channels
        i_hat = ml_detect(r.ravel(), constellation, g.ravel())        # ML detector of Problem 1
        n_errors[k] = count_symbol_errors(i_hat, i)
        energy[k] = np.mean(np.abs(s) ** 2, axis=(0, 1))              # E|s_l|^2 over all channel uses
        gain[k] = np.mean(np.abs(g) ** 2)                             # post-combining SNR / SNR, averaged
    return SchemeResult(n_errors / N_SYMBOLS, n_errors, energy, gain)


# ---------------------------------------------------------------------------
# Discussion numbers
# ---------------------------------------------------------------------------
def summarize(results, curves):
    """Energy check, consistency checks, slopes, array gains and SNR gaps, as printable text."""
    lines = []

    # (a) Total-energy constraint: sum_l E|s_l|^2 = 1 per channel use for every scheme
    lines += ["Average transmitted energy per channel use, measured over all SNR points",
              f"{'scheme':>16}{'scaling':>28}{'total (min / max)':>24}{'per antenna':>18}"]
    for scheme in SCHEMES:
        total = results[scheme.name].energy.sum(axis=1)                    # summed over antennas
        per_antenna = " / ".join(f"{e:.4f}" for e in results[scheme.name].energy.mean(axis=0))
        lines.append(f"{scheme.name:>16}{scheme.scaling:>28}"
                     f"{f'{total.min():.6f} / {total.max():.6f}':>24}{per_antenna:>18}")
    lines.append("")

    # (b) Consistency: single antenna = Problem 1 fading curve; beamforming = MRC with L = 2
    single, bf, mrc = results["Single antenna"].ser, results["Beamforming"].ser, curves["MRC L=2"][0]
    theory = ser_qam4_rayleigh_exact(db_to_linear(SNR_DB))                 # exact 4-QAM Rayleigh SER
    var_theory = theory * (1 - theory) / N_SYMBOLS                         # Var(P_hat) = P(1-P)/N
    var_bf_mrc = (bf * (1 - bf) + mrc * (1 - mrc)) / N_SYMBOLS             # two independent estimates
    lines += ["Consistency checks",
              f"max |single antenna - Problem 1 exact theory| = {np.abs(z_score(single, theory, var_theory)).max():.2f} sigma",
              f"max |beamforming - MRC L=2|                   = {np.abs(z_score(bf, mrc, var_bf_mrc)).max():.2f} sigma",
              ""]
    lines += agreement_lines([(f"{name} vs exact theory", ser, n_err, EXACT_THEORY[name](db_to_linear(SNR_DB)))
                              for name, (ser, n_err) in curves.items()], N_SYMBOLS)
    lines.append("")

    # (c) High-SNR slope of each curve versus the predicted diversity order
    diversity = {"Single antenna": 1, "Beamforming": 2, "Alamouti": 2, "MRC L=2": 2}
    lo, hi = db_to_linear(ASYMPTOTIC_DB)
    lines += [f"High-SNR slope: local slopes over the last {SLOPE_SEGMENTS} steps between points "
              f"with >= {SLOPE_MIN_ERRORS} errors;",
              "least-squares fit over the same points (simulation and exact theory); "
              f"exact theory over {ASYMPTOTIC_DB[0]}-{ASYMPTOTIC_DB[1]} dB",
              f"{'scheme':>16}{'SNR points (dB)':>20}{'local slopes':>24}{'fit sim':>9}"
              f"{'fit exact':>11}{'exact 60-70':>13}{'predicted':>11}"]
    for name, (ser, n_err) in curves.items():
        slopes, snr = high_snr_slopes(SNR_DB, ser, n_err, SLOPE_SEGMENTS, SLOPE_MIN_ERRORS)
        reliable = n_err >= SLOPE_MIN_ERRORS
        fit_sim = loglog_slope(snr, ser[reliable][-(SLOPE_SEGMENTS + 1):])
        fit_exact = loglog_slope(snr, EXACT_THEORY[name](db_to_linear(snr)))
        theory_lo, theory_hi = EXACT_THEORY[name](np.array([lo, hi]))
        asymptotic = -np.log10(theory_hi / theory_lo) / np.log10(hi / lo)
        points = ", ".join(str(v) for v in snr)
        trend = " -> ".join(f"{v:.2f}" for v in slopes)
        lines.append(f"{name:>16}{points:>20}{trend:>24}{fit_sim:>9.2f}{fit_exact:>11.2f}"
                     f"{asymptotic:>13.2f}{diversity[name]:>11}")
    lines.append("")

    # (d) Array gain E|g|^2: 1, E||h||^2 = 2, E||h||^2 / 2 = 1 (MRC L = 2: E||h||^2 = 2)
    array_gain = {"Single antenna": 1, "Beamforming": 2, "Alamouti": 1}
    lines += ["Array gain E|g|^2 (dB): theory, with the value measured in the simulation in brackets",
              f"{'scheme':>16}{'array gain':>18}"]
    for name, theory_gain in array_gain.items():
        measured = 10 * np.log10(results[name].gain.mean())                # averaged over all SNR points
        lines.append(f"{name:>16}{f'{10 * np.log10(theory_gain):.2f} ({format_db(measured)})':>18}")
    lines.append(f"{'MRC L=2':>16}{10 * np.log10(2):>18.2f}   (measured in Problem 2)")
    lines.append("")

    # (e) SNR needed by each scheme, and horizontal gaps between pairs of curves, at fixed SER
    needed = {name: [(snr_at_ser(SNR_DB, *curves[name], t), theory_snr_at_ser(EXACT_THEORY[name], t))
                     for t in GAP_TARGETS] for name in curves}
    header = "".join(f"{f'SER {t:.0e}':>18}" for t in GAP_TARGETS)
    lines += ["SNR (dB) at fixed SER read off the simulated curves (exact theory in brackets)",
              f"{'scheme':>30}{header}"]
    for name, cells in needed.items():
        lines.append(f"{name:>30}" + "".join(f"{f'{format_gap(s)} ({th:.2f})':>18}" for s, th in cells))
    lines += ["", "SNR gap (dB) = SNR of the first scheme - SNR of the second, at the same SER "
              "(exact theory in brackets)", f"{'pair':>30}{header}"]
    pairs = [("Alamouti", "Beamforming"), ("Alamouti", "MRC L=2"),
             ("Beamforming", "MRC L=2"), ("Single antenna", "Alamouti")]
    for a, b in pairs:
        cells = [(sa - sb, ta - tb) for (sa, ta), (sb, tb) in zip(needed[a], needed[b])]
        lines.append(f"{f'{a} - {b}':>30}" + "".join(f"{f'{format_gap(g)} ({th:.2f})':>18}" for g, th in cells))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)                  # seeded once for the whole script
    constellation = qam4()                             # unit-energy 4-QAM from Problem 1

    # 1. The three transmit schemes over the 2 x 1 (or 1 x 1) Rayleigh channel
    results = {scheme.name: simulate_scheme(scheme, constellation, rng) for scheme in SCHEMES}

    # 2. Problem 2 reference: MRC with L = 2 receive antennas, same code as Problem 2
    channel = partial(simo_rayleigh_channel, n_rx=2)
    mrc = simulate_receivers(constellation, SNR_DB, channel, {"MRC": mrc_combine}, N_SYMBOLS, rng)["MRC"]

    # 3. Figure and SER table
    curves = {name: (res.ser, res.n_errors) for name, res in results.items()}
    curves["MRC L=2"] = mrc
    style = {"Single antenna": (PALETTE[0], "o", "-"), "Beamforming": (PALETTE[1], "s", "-"),
             "Alamouti": (PALETTE[2], "D", "-"), "MRC L=2": (PALETTE[3], "^", "--")}   # colour, marker, line
    sims = [SimCurve(name, ser, n_err, *style[name],
                     label="MRC, L = 2 (Problem 2)" if name == "MRC L=2" else name)
            for name, (ser, n_err) in curves.items()]
    at = {(name, t): snr_at_ser(SNR_DB, *curves[name], t) for name in curves for t in (1e-2, 1e-3)}
    arrows = [GapArrow(at["Beamforming", 1e-3], at["Alamouti", 1e-3], 1e-3,          # constant 3 dB
                       f"{at['Alamouti', 1e-3] - at['Beamforming', 1e-3]:.1f} dB", "right"),
              GapArrow(at["Alamouti", 1e-2], at["Single antenna", 1e-2], 1e-2,        # grows as SER falls
                       f"{at['Single antenna', 1e-2] - at['Alamouti', 1e-2]:.1f} dB", "below")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_results(OUT_DIR / "transmit_diversity", SNR_DB, sims, [],
                 title=r"Transmit diversity ($2 \times 1$), 4-QAM, N = $10^6$ symbols per SNR point",
                 legend_loc="lower left", arrows=arrows)

    # 4. Extra figure: the same simulated points (markers) against exact theory (lines)
    markers = [SimCurve(c.name, c.ser, c.n_errors, c.color, c.marker, label=c.label) for c in sims]
    theory_lines = [TheoryCurve(f"{c.name} exact theory", EXACT_THEORY[c.name](db_to_linear(SNR_DB)),
                                c.color, c.linestyle, in_legend=False) for c in sims]
    save_results(OUT_DIR / "transmit_diversity_theory", SNR_DB, markers, theory_lines,
                 title="Transmit diversity: simulation (markers) vs exact theory (lines)",
                 legend_loc="lower left")

    # 5. Numbers for the discussion
    summary = summarize(results, curves)
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)
    print(f"Saved {OUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
