"""Problem 2: receive diversity with MRC and SC over a 1 x L Rayleigh channel, 4-QAM.

For each L in {1, 2, 4}, MRC and SC detect the same symbols, channels and noise.

Run:     python3 -m problem2.receive_diversity
Outputs: results/q2/receive_diversity.{csv,pdf}, results/q2/receive_diversity_theory.{csv,pdf},
         PNG copies in results/q2/images/, results/q2/summary.txt
"""
from functools import partial
from math import factorial
from pathlib import Path

import numpy as np

from common.analysis import (agreement_lines, format_db, format_gap, high_snr_slopes, loglog_slope,
                             snr_at_ser, theory_snr_at_ser, z_score)
from common.channel import complex_gaussian, rayleigh_channel, simo_rayleigh_channel
from common.combining import mrc_combine, selection_combine
from common.constellation import qam4
from common.plotting import PALETTE, GapArrow, SimCurve, TheoryCurve, save_results
from common.simulation import db_to_linear, simulate_receivers, simulate_ser
from common.theory import ser_qam4_mrc, ser_qam4_rayleigh_exact, ser_qam4_sc

SEED = 2026
N_SYMBOLS = 10**6
SNR_DB = np.arange(0, 25, 2)                          # 0, 2, ..., 24 dB
N_RX = (1, 2, 4)                                      # number of receive antennas L
RECEIVERS = {"MRC": mrc_combine, "SC": selection_combine}
EXACT_THEORY = {"MRC": ser_qam4_mrc, "SC": ser_qam4_sc}
LINESTYLE = {"MRC": "-", "SC": "--"}                  # solid for MRC, dashed for SC
MARKER = {"MRC": "o", "SC": "s"}
SLOPE_SEGMENTS, SLOPE_MIN_ERRORS = 3, 50              # slope from the last 3 steps with >= 50 errors
ASYMPTOTIC_DB = (60, 70)                              # SNR range of the asymptotic (theory) slope
GAP_TARGETS = (1e-2, 1e-3, 1e-4)                      # SER levels where the MRC / SC gap is measured
NOISE_CHECK_DB = 10                                   # SNR of the noise-variance check
OUT_DIR = Path(__file__).resolve().parents[1] / "results" / "q2"


def noise_variance_check(constellation, rng):
    """Measured noise variance of r = g x + n for MRC and SC, divided by 1/SNR (should be 1)."""
    snr = db_to_linear(NOISE_CHECK_DB)
    lines = [f"Noise variance of the decision statistic at {NOISE_CHECK_DB} dB, "
             "measured / predicted 1/SNR (expected 1)",
             f"{'L':>4}" + "".join(f"{name:>10}" for name in RECEIVERS)]
    for n_rx in N_RX[1:]:
        x = constellation[rng.integers(len(constellation), size=N_SYMBOLS)]
        h, y = simo_rayleigh_channel(x, snr, rng, n_rx)
        measured = [np.mean(np.abs(r - g * x) ** 2) * snr   # noise n = r - g x
                    for r, g in (combine(y, h) for combine in RECEIVERS.values())]
        lines.append(f"{n_rx:>4}" + "".join(f"{v:>10.4f}" for v in measured))
    return lines


def summarize(results, exact, p1_ser, p1_theory, constellation, rng):
    """Text with the numbers used in the discussion."""
    lines = []

    # L = 1: MRC, SC and Problem 1 should give the same curve
    mrc1, mrc1_err = results["MRC", 1]
    sc1, sc1_err = results["SC", 1]
    lines += ["Consistency check, L = 1 (4-QAM, Rayleigh fading)",
              f"{'SNR_dB':>8}{'MRC L=1':>12}{'SC L=1':>12}{'Problem 1 sim':>16}{'P1 exact theory':>18}"]
    lines += [f"{s:>8}{a:>12.4g}{b:>12.4g}{c:>16.4g}{d:>18.4g}"
              for s, a, b, c, d in zip(SNR_DB, mrc1, sc1, p1_ser, p1_theory)]
    var_theory = p1_theory * (1 - p1_theory) / N_SYMBOLS
    var_two_sims = (mrc1 * (1 - mrc1) + p1_ser * (1 - p1_ser)) / N_SYMBOLS
    lines += [f"MRC and SC (L = 1) error counts identical at every SNR: {np.array_equal(mrc1_err, sc1_err)}",
              f"max |MRC L=1 - Problem 1 sim|    = {np.abs(z_score(mrc1, p1_ser, var_two_sims)).max():.2f} sigma",
              f"max |MRC L=1 - P1 exact theory|  = {np.abs(z_score(mrc1, p1_theory, var_theory)).max():.2f} sigma",
              ""]

    # High-SNR slope of each curve (should approach the diversity order L)
    lo, hi = db_to_linear(ASYMPTOTIC_DB)
    lines += [f"High-SNR slope: local slopes over the last {SLOPE_SEGMENTS} steps between points "
              f"with >= {SLOPE_MIN_ERRORS} errors;",
              "least-squares fit over the same points (simulation and exact theory); "
              f"exact theory over {ASYMPTOTIC_DB[0]}-{ASYMPTOTIC_DB[1]} dB",
              f"{'receiver':>10}{'L':>4}{'SNR points (dB)':>20}{'local slopes':>24}{'fit sim':>9}"
              f"{'fit exact':>11}{'exact 60-70':>13}{'predicted':>11}"]
    for (name, n_rx), (ser, n_err) in results.items():
        slopes, snr = high_snr_slopes(SNR_DB, ser, n_err, SLOPE_SEGMENTS, SLOPE_MIN_ERRORS)
        reliable = n_err >= SLOPE_MIN_ERRORS
        fit_sim = loglog_slope(snr, ser[reliable][-(SLOPE_SEGMENTS + 1):])
        fit_exact = loglog_slope(snr, EXACT_THEORY[name](db_to_linear(snr), n_rx))
        theory = EXACT_THEORY[name](np.array([lo, hi]), n_rx)
        asymptotic = -np.log10(theory[1] / theory[0]) / np.log10(hi / lo)
        points = ", ".join(str(v) for v in snr)
        trend = " -> ".join(f"{v:.2f}" for v in slopes)
        lines.append(f"{name:>10}{n_rx:>4}{points:>20}{trend:>24}{fit_sim:>9.2f}"
                     f"{fit_exact:>11.2f}{asymptotic:>13.2f}{n_rx:>11}")
    lines.append("")

    # Array gain: E||h||^2 = L for MRC, E[max |h_l|^2] = 1 + 1/2 + ... + 1/L for SC
    lines += ["Array gain [2, Sec. 3.4] (dB): theory, with a Monte Carlo estimate in brackets",
              f"{'L':>4}{'MRC':>18}{'SC':>18}{'MRC - SC':>12}"]
    for n_rx in N_RX:
        mrc_db = 10 * np.log10(n_rx)
        sc_db = 10 * np.log10(sum(1 / k for k in range(1, n_rx + 1)))
        gains = np.abs(complex_gaussian((N_SYMBOLS, n_rx), 1.0, rng)) ** 2
        mrc_mc = 10 * np.log10(np.mean(np.sum(gains, axis=1)))
        sc_mc = 10 * np.log10(np.mean(np.max(gains, axis=1)))
        lines.append(f"{n_rx:>4}{f'{mrc_db:.2f} ({format_db(mrc_mc)})':>18}"
                     f"{f'{sc_db:.2f} ({format_db(sc_mc)})':>18}{mrc_db - sc_db:>12.2f}")
    lines.append("")

    # SNR needed by MRC and SC at the target SER levels, and the gap between them
    header = "".join(f"{f'SER {t:.0e}':>16}" for t in GAP_TARGETS)
    lines += ["SNR (dB) at fixed SER read off the simulated curves (exact theory in brackets);",
              "high-SNR limit of the gap SC - MRC: 10 log10(L!) / L",
              f"{'':>10}{header}{'limit':>10}"]
    for n_rx in N_RX:
        needed = {name: [(snr_at_ser(SNR_DB, *results[name, n_rx], t),
                          theory_snr_at_ser(partial(EXACT_THEORY[name], n_rx=n_rx), t)) for t in GAP_TARGETS]
                  for name in RECEIVERS}
        for name in RECEIVERS:
            cells = "".join(f"{f'{format_gap(s)} ({th:.2f})':>16}" for s, th in needed[name])
            lines.append(f"{f'{name} L={n_rx}':>10}{cells}")
        gaps = "".join(f"{f'{format_gap(sc - mrc)} ({sc_th - mrc_th:.2f})':>16}"
                       for (mrc, mrc_th), (sc, sc_th) in zip(needed["MRC"], needed["SC"]))
        lines.append(f"{f'gap L={n_rx}':>10}{gaps}{10 * np.log10(factorial(n_rx)) / n_rx:>10.2f}")
    lines.append("")

    # Agreement of every curve with its exact theory
    lines += agreement_lines([(f"{name} L={n_rx} vs exact theory", ser, n_err, exact[name, n_rx])
                              for (name, n_rx), (ser, n_err) in results.items()], N_SYMBOLS)
    lines.append("")

    # Noise variance of the decision statistics (Task 1)
    lines += noise_variance_check(constellation, rng)
    return "\n".join(lines) + "\n"


def main():
    rng = np.random.default_rng(SEED)   # seed set once for the whole script
    constellation = qam4()
    snr = db_to_linear(SNR_DB)

    # 1. Simulation: for each L, MRC and SC use the same symbols, channels and noise
    runs = {}
    for n_rx in N_RX:
        channel = partial(simo_rayleigh_channel, n_rx=n_rx)
        runs[n_rx] = simulate_receivers(constellation, SNR_DB, channel, RECEIVERS, N_SYMBOLS, rng)
    results = {(name, n_rx): runs[n_rx][name]
               for name in RECEIVERS for n_rx in N_RX}
    exact = {(name, n_rx): EXACT_THEORY[name](snr, n_rx) for name, n_rx in results}

    # 2. Problem 1 reference: single-antenna fading curve (simulation and exact theory)
    p1_ser, _ = simulate_ser(constellation, SNR_DB, rayleigh_channel, N_SYMBOLS, rng)
    p1_theory = ser_qam4_rayleigh_exact(snr)

    # 3. Main figure: one colour per L, solid MRC, dashed SC, arrows for the MRC / SC gap
    def style(name, n_rx):
        return PALETTE[N_RX.index(n_rx)], MARKER[name]

    sims = [SimCurve(f"{name} L={n_rx}", ser, n_err, *style(name, n_rx), LINESTYLE[name],
                     label=f"{name}, L = {n_rx}")
            for (name, n_rx), (ser, n_err) in results.items()]
    reference = TheoryCurve("Problem 1 (exact theory)", p1_theory,
                            color="#c4c4c4", linewidth=8)   # grey band behind the L = 1 curves
    arrows = []
    for n_rx, side in ((2, "right"), (4, "left")):   # label placed where the plot is empty
        mrc_db, sc_db = (snr_at_ser(SNR_DB, *results[name, n_rx], 1e-3) for name in RECEIVERS)
        arrows.append(GapArrow(mrc_db, sc_db, 1e-3, f"L = {n_rx}: {sc_db - mrc_db:.1f} dB", side))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_results(OUT_DIR / "receive_diversity", SNR_DB, sims, [reference],
                 title="Receive diversity (MRC vs SC), 4-QAM, N = $10^6$ symbols per SNR point",
                 legend_loc="lower left", arrows=arrows)

    # 4. Extra figure: the same points against exact theory
    markers = [SimCurve(c.name, c.ser, c.n_errors, c.color, c.marker, label=c.label) for c in sims]
    theory_lines = [TheoryCurve(f"{name} L={n_rx} exact theory", exact[name, n_rx], *style(name, n_rx)[:1],
                                LINESTYLE[name], in_legend=False) for name, n_rx in results]
    save_results(OUT_DIR / "receive_diversity_theory", SNR_DB, markers, theory_lines,
                 title="Receive diversity: simulation (markers) vs exact theory (lines)",
                 legend_loc="lower left")

    # 5. Summary numbers used in the report
    summary = summarize(results, exact, p1_ser, p1_theory, constellation, rng)
    (OUT_DIR / "summary.txt").write_text(summary)
    print(summary)
    print(f"Saved {OUT_DIR / 'summary.txt'}")


if __name__ == "__main__":
    main()
