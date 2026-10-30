"""Problem 3: transmit diversity over a 2 x 1 Rayleigh channel, 4-QAM.

Schemes: single antenna, transmit beamforming and the Alamouti code, compared with
MRC (L = 2) from Problem 2. Every scheme sends a total energy of 1 per channel use.

Run:     python3 -m problem3.transmit_diversity
Outputs: results/q3/transmit_diversity.{csv,pdf}, results/q3/transmit_diversity_theory.{csv,pdf},
         PNG copies in results/q3/images/, results/q3/summary.txt
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
N_SYMBOLS = 10**6                                     # symbols per SNR point (Alamouti: N/2 blocks of 2)
SNR_DB = np.arange(0, 25, 2)                          # 0, 2, ..., 24 dB
SLOPE_SEGMENTS, SLOPE_MIN_ERRORS = 3, 50              # slope from the last 3 steps with >= 50 errors
ASYMPTOTIC_DB = (60, 70)                              # SNR range of the asymptotic (theory) slope
GAP_TARGETS = (1e-2, 1e-3, 1e-4)                      # SER levels where the gaps are measured
EXACT_THEORY = {                                      # exact 4-QAM SER of each curve
    "Single antenna": ser_qam4_rayleigh_exact,
    "Beamforming": partial(ser_qam4_mrc, n_rx=2),     # same as MRC with L = 2
    "Alamouti": ser_qam4_alamouti,
    "MRC L=2": partial(ser_qam4_mrc, n_rx=2),
}
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "q3"


# Transmitters return s with shape (blocks, channel uses per block, antennas).

def single_antenna_tx(x, h):
    """Single antenna: send x as it is."""
    return x[..., np.newaxis]


def beamforming_tx(x, h):
    """Beamforming: both antennas send v x with v = h* / ||h||, so ||v|| = 1."""
    v = np.conj(h) / np.linalg.norm(h, axis=-1, keepdims=True)
    return x[..., np.newaxis] * v[:, np.newaxis, :]


def alamouti_tx(x, h):
    """Alamouti: S = (1/sqrt(2)) [[x1, x2], [-x2*, x1*]], rows = channel uses, columns = antennas.

    The transmitter does not use h.
    """
    x1, x2 = x[:, 0], x[:, 1]
    use1 = np.stack([x1, x2], axis=-1)                                # channel use 1: antennas send x1, x2
    use2 = np.stack([-np.conj(x2), np.conj(x1)], axis=-1)             # channel use 2: antennas send -x2*, x1*
    return np.stack([use1, use2], axis=1) / np.sqrt(2)                # 1/sqrt(2): total energy 1 per channel use


# Receivers return (r, g): one scalar channel r = g x + n for every symbol.

def beamforming_rx(y, h):
    """Beamforming: y = ||h|| x + w, so the gain is ||h||."""
    return y, np.linalg.norm(h, axis=-1, keepdims=True)


def alamouti_rx(y, h):
    """Alamouti combining: r_k = (||h|| / sqrt(2)) x_k + n_k for both symbols of a block."""
    h1, h2 = h[:, 0], h[:, 1]
    y1, y2 = y[:, 0], y[:, 1]
    h_norm = np.linalg.norm(h, axis=-1)
    r1 = (np.conj(h1) * y1 + h2 * np.conj(y2)) / h_norm               # x2 cancels, x1 remains
    r2 = (np.conj(h2) * y1 - h1 * np.conj(y2)) / h_norm               # x1 cancels, x2 remains
    gain = h_norm / np.sqrt(2)
    return np.stack([r1, r2], axis=-1), np.stack([gain, gain], axis=-1)


@dataclass
class MisoScheme:
    """One transmission scheme."""
    name: str
    n_tx: int                  # transmit antennas
    block_len: int             # channel uses per block (h fixed within a block)
    transmit: Callable
    receive: Callable
    scaling: str               # text for the summary


SCHEMES = [
    MisoScheme("Single antenna", 1, 1, single_antenna_tx, no_combining, "s = x"),
    MisoScheme("Beamforming", 2, 1, beamforming_tx, beamforming_rx, "s_l = (h_l* / ||h||) x"),
    MisoScheme("Alamouti", 2, 2, alamouti_tx, alamouti_rx, "s_l = +-x_k(*) / sqrt(2)"),
]


@dataclass
class SchemeResult:
    """Results of one scheme, one value per SNR point."""
    ser: np.ndarray
    n_errors: np.ndarray
    energy: np.ndarray         # average |s_l|^2 on each antenna
    gain: np.ndarray           # average |g|^2 (array gain)


def simulate_scheme(scheme, constellation, rng):
    """Monte Carlo SER of one scheme, plus the transmitted energy and the array gain."""
    n_blocks = N_SYMBOLS // scheme.block_len
    n_errors = np.zeros(len(SNR_DB), dtype=int)
    energy = np.zeros((len(SNR_DB), scheme.n_tx))
    gain = np.zeros(len(SNR_DB))
    for k, snr in enumerate(db_to_linear(SNR_DB)):
        i = rng.integers(len(constellation), size=N_SYMBOLS)
        x = constellation[i].reshape(n_blocks, scheme.block_len)      # group the symbols into blocks
        h = complex_gaussian((n_blocks, scheme.n_tx), 1.0, rng)       # one channel per block
        s = scheme.transmit(x, h)
        y = miso_channel(s, h, snr, rng)
        r, g = scheme.receive(y, h)
        i_hat = ml_detect(r.ravel(), constellation, g.ravel())
        n_errors[k] = count_symbol_errors(i_hat, i)
        energy[k] = np.mean(np.abs(s) ** 2, axis=(0, 1))              # energy sent on each antenna
        gain[k] = np.mean(np.abs(g) ** 2)
    return SchemeResult(n_errors / N_SYMBOLS, n_errors, energy, gain)


def summarize(results, curves):
    """Text with the numbers used in the discussion."""
    lines = []

    # Energy check: total energy per channel use should be 1 for every scheme
    lines += ["Average transmitted energy per channel use, measured over all SNR points",
              f"{'scheme':>16}{'scaling':>28}{'total (min / max)':>24}{'per antenna':>18}"]
    for scheme in SCHEMES:
        total = results[scheme.name].energy.sum(axis=1)
        per_antenna = " / ".join(f"{e:.4f}" for e in results[scheme.name].energy.mean(axis=0))
        lines.append(f"{scheme.name:>16}{scheme.scaling:>28}"
                     f"{f'{total.min():.6f} / {total.max():.6f}':>24}{per_antenna:>18}")
    lines.append("")

    # Consistency: single antenna = Problem 1, beamforming = MRC (L = 2), all curves vs exact theory
    single, bf, mrc = results["Single antenna"].ser, results["Beamforming"].ser, curves["MRC L=2"][0]
    theory = ser_qam4_rayleigh_exact(db_to_linear(SNR_DB))
    var_theory = theory * (1 - theory) / N_SYMBOLS
    var_bf_mrc = (bf * (1 - bf) + mrc * (1 - mrc)) / N_SYMBOLS
    lines += ["Consistency checks",
              f"max |single antenna - Problem 1 exact theory| = {np.abs(z_score(single, theory, var_theory)).max():.2f} sigma",
              f"max |beamforming - MRC L=2|                   = {np.abs(z_score(bf, mrc, var_bf_mrc)).max():.2f} sigma",
              ""]
    lines += agreement_lines([(f"{name} vs exact theory", ser, n_err, EXACT_THEORY[name](db_to_linear(SNR_DB)))
                              for name, (ser, n_err) in curves.items()], N_SYMBOLS)
    lines.append("")

    # High-SNR slope of each curve (should approach the diversity order)
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

    # Array gain E|g|^2: 1 (single antenna), 2 (beamforming), 1 (Alamouti)
    array_gain = {"Single antenna": 1, "Beamforming": 2, "Alamouti": 1}
    lines += ["Array gain E|g|^2 (dB): theory, with the value measured in the simulation in brackets",
              f"{'scheme':>16}{'array gain':>18}"]
    for name, theory_gain in array_gain.items():
        measured = 10 * np.log10(results[name].gain.mean())
        lines.append(f"{name:>16}{f'{10 * np.log10(theory_gain):.2f} ({format_db(measured)})':>18}")
    lines.append(f"{'MRC L=2':>16}{10 * np.log10(2):>18.2f}   (measured in Problem 2)")
    lines.append("")

    # SNR needed at the target SER levels, and the gaps between schemes
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


def main():
    rng = np.random.default_rng(SEED)   # seed set once for the whole script
    constellation = qam4()

    # 1. Simulate the three transmit schemes
    results = {scheme.name: simulate_scheme(scheme, constellation, rng) for scheme in SCHEMES}

    # 2. MRC with L = 2, using the Problem 2 code
    channel = partial(simo_rayleigh_channel, n_rx=2)
    mrc = simulate_receivers(constellation, SNR_DB, channel, {"MRC": mrc_combine}, N_SYMBOLS, rng)["MRC"]

    # 3. Main figure, with arrows for two of the gaps
    curves = {name: (res.ser, res.n_errors) for name, res in results.items()}
    curves["MRC L=2"] = mrc
    style = {"Single antenna": (PALETTE[0], "o", "-"), "Beamforming": (PALETTE[1], "s", "-"),
             "Alamouti": (PALETTE[2], "D", "-"), "MRC L=2": (PALETTE[3], "^", "--")}   # colour, marker, line
    sims = [SimCurve(name, ser, n_err, *style[name],
                     label="MRC, L = 2 (Problem 2)" if name == "MRC L=2" else name)
            for name, (ser, n_err) in curves.items()]
    at = {(name, t): snr_at_ser(SNR_DB, *curves[name], t) for name in curves for t in (1e-2, 1e-3)}
    arrows = [GapArrow(at["Beamforming", 1e-3], at["Alamouti", 1e-3], 1e-3,
                       f"{at['Alamouti', 1e-3] - at['Beamforming', 1e-3]:.1f} dB", "right"),
              GapArrow(at["Alamouti", 1e-2], at["Single antenna", 1e-2], 1e-2,
                       f"{at['Single antenna', 1e-2] - at['Alamouti', 1e-2]:.1f} dB", "below")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_results(OUT_DIR / "transmit_diversity", SNR_DB, sims, [],
                 title=r"Transmit diversity ($2 \times 1$), 4-QAM, N = $10^6$ symbols per SNR point",
                 legend_loc="lower left", arrows=arrows)

    # 4. Extra figure: the same points against exact theory
    markers = [SimCurve(c.name, c.ser, c.n_errors, c.color, c.marker, label=c.label) for c in sims]
    theory_lines = [TheoryCurve(f"{c.name} exact theory", EXACT_THEORY[c.name](db_to_linear(SNR_DB)),
                                c.color, c.linestyle, in_legend=False) for c in sims]
    save_results(OUT_DIR / "transmit_diversity_theory", SNR_DB, markers, theory_lines,
                 title="Transmit diversity: simulation (markers) vs exact theory (lines)",
                 legend_loc="lower left")

    # 5. Summary numbers used in the report
    summary = summarize(results, curves)
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)
    print(f"Saved {OUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
